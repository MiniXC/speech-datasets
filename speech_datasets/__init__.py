import os, errno
import sys
from pathlib import Path
import torch
from torch import nn
import numpy as np
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from PIL import Image
from vocex import Vocex
import json

# for hashing
from hashlib import sha256

from simple_hifigan import Synthesiser


def resample_nearest(array, new_length):
    indices = np.linspace(0, len(array) - 1, new_length)
    nearest_indices = np.rint(indices).astype(int)
    return array[nearest_indices]


def fill_sequence(arr):
    non_zero_indices = np.where(arr != 0)[0]

    for i in range(len(non_zero_indices) - 1):
        left_idx = non_zero_indices[i]
        right_idx = non_zero_indices[i + 1]
        midpoint = (left_idx + right_idx) // 2

        arr[left_idx:midpoint] = arr[left_idx]
        arr[midpoint:right_idx] = arr[right_idx]

    # For the parts of the array before the first non-zero and after the last non-zero
    if non_zero_indices[0] > 0:
        arr[: non_zero_indices[0]] = arr[non_zero_indices[0]]
    if non_zero_indices[-1] < len(arr) - 1:
        arr[non_zero_indices[-1] + 1 :] = arr[non_zero_indices[-1]]

    return arr


def resample(x, vpw=5):
    return np.interp(np.linspace(0, 1, vpw), np.linspace(0, 1, len(x)), x)


def locality_sensitive_hashing(x, hyperplanes):
    return torch.matmul(x, hyperplanes.T)


def repeated_phones_to_phones_and_lengths(phones):
    phone_list = []
    phone_lengths = []
    current_phone = phones[0]
    current_length = 1
    for i in range(1, len(phones)):
        if phones[i] == current_phone:
            current_length += 1
        else:
            phone_list.append(current_phone)
            phone_lengths.append(current_length)
            current_phone = phones[i]
            current_length = 1
    phone_list.append(current_phone)
    phone_lengths.append(current_length)
    phone_list = np.array(phone_list)
    phone_lengths = np.array(phone_lengths)
    return phone_list, phone_lengths


def to_img(x, min_val=0, max_val=1):
    x = torch.from_numpy(x)
    x = torch.clamp(x, min=min_val, max=max_val)
    x = (x - min_val) / (max_val - min_val)
    x = x * 255
    x = x.type(torch.uint8)
    x = x.flip(1)
    return x


def _pop(x, idx):
    return [x[i] for i in range(len(x)) if i != idx]


class Preprocessor:
    def __init__(self, target_location, device=None, allow_overwrite=False):
        self.phone_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        )
        self.phone_model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        ).to(device)
        self.id2phone = self.phone_processor.tokenizer.decoder
        self.id2phone[len(self.id2phone)] = "<s2>"
        self.id2phone[len(self.id2phone)] = "<s1>"
        self.id2phone[len(self.id2phone)] = "<s0>"
        self.synthesiser = Synthesiser()
        self.vocex = Vocex.from_pretrained("cdminix/vocex")
        self.vocex.model.eval()
        self.device = device
        self.vocex.model.to(self.device)
        self.target_location = target_location
        # make target location if it doesn't exist
        Path(self.target_location).mkdir(parents=True, exist_ok=True)
        # check if empty
        if len(os.listdir(self.target_location)) != 0 and not allow_overwrite:
            raise ValueError(
                "Target location is not empty, please empty it before running"
            )
        self.target_location = Path(self.target_location)
        # save id2phone
        with open(self.target_location / "id2phone.json", "w") as f:
            json.dump(self.id2phone, f)
        self.pitch_range = (50, 500)
        self.energy_range = (0, 0.5)
        self.vad_range = (0, 1)
        self.speaker_range = (-200, 200)
        self.length_range = (0, 11)
        self.mel_range = (-11, 2)
        self.n_planes = 40
        # torch.random.manual_seed(0)
        self.hyperplanes = torch.randn((self.n_planes, 256)).to(self.device)
        model, utils = torch.hub.load(
            repo_or_dir="snakers4/silero-vad", model="silero_vad", force_reload=True
        )
        (
            get_speech_timestamps,
            _,
            _,
            _,
            _,
        ) = utils
        self.get_speech_timestamps = get_speech_timestamps
        self.vad_model = model
        self.vad_model.to(self.device)

    def __call__(self, batch):
        batched_audio = []
        batched_audio16 = []
        batched_speaker = []
        batched_speaker_global = []
        book_dirs = []
        hashes = []
        new_batch = []
        for b in batch:
            filename = Path(b["id"])
            speaker = str(b["speaker"])
            if "_" in filename.name:
                chapter = filename.name.split("_")[1]
            else:
                chapter = filename.name.split("-")[1]

            directory = self.target_location / speaker / chapter
            directory.mkdir(parents=True, exist_ok=True)

            # check if item has already been processed
            if (directory / f"{filename.stem}_mel.png").exists():
                print(f"Skipping {filename.stem}")
                continue

            new_batch.append(b)
            book_dirs.append(directory)

            hashes.append(filename.stem)

            audio, sr = torchaudio.load(b["audio"])
            audio = audio / torch.max(torch.abs(audio))
            if sr != 16000:
                audio16 = torchaudio.transforms.Resample(sr, 16000)(audio)
            else:
                audio16 = audio

            batched_audio.append(audio.squeeze(0))
            batched_audio16.append(audio16.squeeze(0))

        # return if batch is empty
        if len(batched_audio) == 0:
            return batch

        batched_audio = nn.utils.rnn.pad_sequence(
            batched_audio, batch_first=True, padding_value=0
        )

        mels, mels_mask = self.synthesiser.wavs_to_mel(batched_audio.float(), sr=sr)
        mels = [mels[i].T[mels_mask[i]] for i in range(len(mels))]

        all_silences = []
        for j, audio16 in enumerate(batched_audio16):
            # add half a second of silence to the start and end
            audio16 = torch.cat([torch.zeros(8000), audio16, torch.zeros(8000)], dim=0)
            with torch.no_grad():
                vad_result = self.get_speech_timestamps(
                    audio16.to(self.device), self.vad_model, sampling_rate=16000
                )
            vad_factor = mels[j].shape[0] / (len(audio16) - 16000)
            max_len = mels[j].shape[0]
            vad_result = [
                (
                    int(np.ceil((v["start"] - 8000) * vad_factor)),
                    int(np.ceil((v["end"] - 8000) * vad_factor)),
                )
                for v in vad_result
            ]
            silences = []
            if len(vad_result) > 0:
                if vad_result[0][0] != 0:
                    silences.append((0, vad_result[0][0]))
                for i in range(len(vad_result) - 1):
                    silences.append((vad_result[i][1], vad_result[i + 1][0]))
                if vad_result[-1][1] <= max_len:
                    silences.append((vad_result[-1][1], max_len))
            all_silences.append(silences)

        predicted_ids = []
        for i in range(len(batched_audio16)):
            input_values = self.phone_processor(
                batched_audio16[i],
                return_tensors="pt",
                sampling_rate=16000,
            ).input_values
            with torch.no_grad():
                logits = self.phone_model(input_values.to(self.device)).logits.cpu()
            ids = torch.argmax(logits, dim=-1)[0].numpy()
            ids = resample_nearest(ids, mels[i].shape[0])
            predicted_ids.append(ids)

        skip_idxs = []

        for i in range(len(predicted_ids)):
            id_factor = mels[i].shape[0] / len(predicted_ids[i])
            # add silence tokens
            ms_per_frame = 11.5
            for silence_start, silence_end in all_silences[i]:
                if (silence_end - silence_start) * ms_per_frame < 150:
                    predicted_ids[i][silence_start:silence_end] = len(self.id2phone) - 1
                elif (silence_end - silence_start) * ms_per_frame < 500:
                    predicted_ids[i][silence_start:silence_end] = len(self.id2phone) - 2
                else:
                    predicted_ids[i][silence_start:silence_end] = len(self.id2phone) - 3
            try:
                predicted_ids[i] = fill_sequence(predicted_ids[i])
            except IndexError:
                skip_idxs.append(i)

        batched_speaker = []
        batched_speaker_global = []

        new_batch = [new_batch[i] for i in range(len(new_batch)) if i not in skip_idxs]

        for i in range(len(new_batch)):
            save_dir = book_dirs[i]
            utt_hash = hashes[i]
            # write text
            with open(save_dir / f"{utt_hash}_text.txt", "w") as f:
                f.write(batch[i]["text"])
            # vocex
            with torch.no_grad():
                if len(mels[i]) <= 512:
                    vocex_results = self.vocex.model(
                        mels[i].unsqueeze(0).to(self.device), inference=True
                    )
                else:
                    mel_chunks = torch.split(mels[i], 512, dim=0)
                    chunk_results = []
                    for chunk in mel_chunks:
                        chunk_results.append(
                            self.vocex.model(
                                chunk.unsqueeze(0).to(self.device), inference=True
                            )
                        )
                    vocex_results = {
                        "measures": {},
                        "dvector": [],
                        "dvector_time": [],
                    }
                    for key in chunk_results[0]["measures"].keys():
                        vocex_results["measures"][key] = torch.cat(
                            [i["measures"][key] for i in chunk_results],
                            axis=1,
                        )
                    vocex_results["dvector"] = torch.mean(
                        torch.cat([i["dvector"] for i in chunk_results]), dim=0
                    ).unsqueeze(0)
                    vocex_results["dvector_time"] = torch.cat(
                        [i["dvector_time"] for i in chunk_results], axis=1
                    )
            pitch = vocex_results["measures"]["pitch"][0]
            pitch = torch.clamp(pitch, self.pitch_range[0], self.pitch_range[1])
            pitch = (pitch - self.pitch_range[0]) / (
                self.pitch_range[1] - self.pitch_range[0]
            )
            energy = vocex_results["measures"]["energy"][0]
            energy = torch.clamp(energy, self.energy_range[0], self.energy_range[1])
            energy = (energy - self.energy_range[0]) / (
                self.energy_range[1] - self.energy_range[0]
            )
            phone_list, phone_lengths = repeated_phones_to_phones_and_lengths(
                predicted_ids[i]
            )

            # phones
            np.save(save_dir / f"{utt_hash}_phones.npy", phone_list.astype(np.ushort))
            # phone lengths
            np.save(
                save_dir / f"{utt_hash}_phone_lengths.npy",
                phone_lengths.astype(np.ushort),
            )

            vad = vocex_results["measures"]["voice_activity_binary"][0]
            vad = torch.clamp(vad, self.vad_range[0], self.vad_range[1])
            vad = (vad - self.vad_range[0]) / (self.vad_range[1] - self.vad_range[0])
            dvec = vocex_results["dvector"][0]
            dvec_time = vocex_results["dvector_time"][0]
            dvec_time = locality_sensitive_hashing(dvec_time, self.hyperplanes)
            dvec_time = torch.clamp(
                dvec_time, self.speaker_range[0], self.speaker_range[1]
            )
            dvec_time = (dvec_time - self.speaker_range[0]) / (
                self.speaker_range[1] - self.speaker_range[0]
            )
            batched_speaker.append(dvec_time.cpu().numpy())
            batched_speaker_global.append(dvec.cpu().numpy())
            current_idx = 0
            vals_per_window = 10
            prosody = np.zeros((len(phone_lengths), vals_per_window * 3 + 1))
            speaker = np.zeros((len(phone_lengths), self.n_planes))
            for j, d in enumerate(phone_lengths):
                d = int(d)
                if d == 0:
                    prosody[j, :] = 0
                    continue
                if current_idx + d > len(pitch):
                    continue
                pitch_window = pitch[current_idx : current_idx + d].cpu()
                energy_window = energy[current_idx : current_idx + d].cpu()
                va_window = vad[current_idx : current_idx + d].cpu()
                prosody[j, 1 : vals_per_window + 1] = resample(
                    pitch_window, vals_per_window
                )
                prosody[j, vals_per_window + 1 : vals_per_window * 2 + 1] = resample(
                    energy_window, vals_per_window
                )
                prosody[
                    j, vals_per_window * 2 + 1 : vals_per_window * 3 + 1
                ] = resample(va_window, vals_per_window)
                speaker[j, :] = np.mean(
                    batched_speaker[i][current_idx : current_idx + d],
                    axis=0,
                )
                current_idx += d
            lengths = np.array(phone_lengths)
            lengths = np.log2(lengths)
            lengths = np.clip(lengths, self.length_range[0], self.length_range[1])
            lengths = lengths / (self.length_range[1] - self.length_range[0])
            # lengths_denorm = lengths * (self.length_range[1] - self.length_range[0])
            # lengths_denorm = 2**lengths_denorm
            # assert int(lengths_denorm.sum()) == mels[i].shape[0]
            prosody[:, 0] = lengths
            prosody = to_img(prosody, min_val=0, max_val=1)
            prosody = Image.fromarray(prosody.numpy().T)
            prosody.save(save_dir / f"{utt_hash}_prosody.png")
            # speaker
            speaker = to_img(speaker, min_val=0, max_val=1)
            speaker = Image.fromarray(speaker.numpy().T)
            speaker.save(save_dir / f"{utt_hash}_speaker.png")
            ## overall speaker
            overall_speaker = batched_speaker_global[i]
            np.save(save_dir / f"{utt_hash}_speaker.npy", overall_speaker)
            # mel
            mel = mels[i].numpy()
            mel = np.clip(mel, self.mel_range[0], self.mel_range[1])
            mel = (mel - self.mel_range[0]) / (self.mel_range[1] - self.mel_range[0])
            mel = to_img(mel, min_val=0, max_val=1)
            mel = Image.fromarray(mel.numpy().T)
            mel.save(save_dir / f"{utt_hash}_mel.png")

        return batch
