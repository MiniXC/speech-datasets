import os
import sys
from pathlib import Path
import torch
from torch import nn
import numpy as np
import torchaudio
from transformers import Wav2Vec2Processor, Wav2Vec2ForCTC
from scipy import signal
from PIL import Image
from vocex import Vocex
import humanhash
import json

# for hashing
from hashlib import sha256

from simple_hifigan import Synthesiser

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
        arr[:non_zero_indices[0]] = arr[non_zero_indices[0]]
    # this is removed due to CTC quirks
    # if non_zero_indices[-1] < len(arr) - 1:
    #     arr[non_zero_indices[-1]+1:] = arr[non_zero_indices[-1]]

    return arr

def resample(x, vpw=5):
    return np.interp(np.linspace(0, 1, vpw), np.linspace(0, 1, len(x)), x)

def locality_sensitive_hashing(x, hyperplanes):
    return torch.matmul(x, hyperplanes.T)


def zero_split(ids):
    non_zeros = np.where(ids != 0)[0]
    diffs = np.diff(non_zeros)
    lengths = np.ones_like(non_zeros, dtype=int)
    for i in range(len(diffs)):
        extra_zeros = diffs[i] - 1
        lengths[i] += extra_zeros // 2
        lengths[i + 1] += extra_zeros - extra_zeros // 2
    return ids[non_zeros], lengths


def resample_np(sequence, new_length):
    indices = np.linspace(0, len(sequence) - 1, new_length)
    rounded_indices = np.round(indices).astype(int)
    resampled = sequence[rounded_indices]
    return resampled


def flatten_list(l):
    return [item for sublist in l for item in sublist]


def drc(x, C=1, clip_val=1e-7):
    return torch.log(torch.clamp(x, min=clip_val) * C)


def to_img(x, min_val=0, max_val=1):
    x = torch.from_numpy(x)
    x = torch.clamp(x, min=min_val, max=max_val)
    x = (x - min_val) / (max_val - min_val)
    x = x * 255
    x = x.type(torch.uint8)
    x = x.flip(1)
    return x


def from_img(x, min_val=0, max_val=1):
    x = torch.from_numpy(x)
    x = x.type(torch.float32)
    x = x / 255
    x = x * (max_val - min_val)
    x = x + min_val
    x = x.flip(1)
    return x


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
        self.pitch_range = (50, 300)
        self.energy_range = (0, 0.2)
        self.vad_range = (0, 1)
        self.speaker_range = (-200, 200)
        self.length_range = (0, 8)
        self.mel_range = (-11, 2)
        self.n_planes = 40
        torch.random.manual_seed(0)
        self.hyperplanes = torch.randn((self.n_planes, 256)).to(self.device)

    def __call__(self, batch):
        batched_audio = []
        batched_audio16 = []
        batched_speaker = []
        batched_speaker_global = []
        hashes = []
        spk_dirs = []
        book_dirs = []
        for b in batch:
            # hash the name of the file
            h = sha256()
            h.update(b["id"].encode())
            h = h.hexdigest()
            file_hash = h
            # check if item has already been processed
            if (self.target_location / f"{file_hash}_text.txt").exists():
                continue
            hashes.append(h)
            # hash the speaker
            h = sha256()
            h.update(str(b["speaker_id"]).encode())
            h = h.hexdigest()
            h = humanhash.humanize(h).replace("-", "_")
            spk_dir = self.target_location / h
            spk_dir.mkdir(parents=True, exist_ok=True)
            spk_dirs.append(spk_dir)
            # book
            book = str(b["chapter_id"])
            book_hash = sha256()
            book_hash.update(book.encode())
            book_hash = book_hash.hexdigest()
            book_dir = spk_dir / book_hash
            book_dir.mkdir(parents=True, exist_ok=True)
            book_dirs.append(book_dir)
            # transcript
            text = b["text"]
            with open(book_dir / f"{file_hash}_text.txt", "w") as f:
                f.write(text)

            if "audio" not in b:
                audio, sr = torchaudio.load(b["audio_path"])
            else:
                audio = torch.tensor(b["audio"]["array"])
                sr = b["audio"]["sampling_rate"]
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
        batched_audio16 = nn.utils.rnn.pad_sequence(
            batched_audio16, batch_first=True, padding_value=0
        )

        input_values = self.phone_processor(
            batched_audio16.tolist(), return_tensors="pt", sampling_rate=16000
        ).input_values

        # retrieve logits
        with torch.no_grad():
            logits = self.phone_model(input_values.to(self.device)).logits.cpu()

        mels, mels_mask = self.synthesiser.wavs_to_mel(batched_audio.float(), sr=sr)
        # fp16
        # mels = mels.half()

        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)

        # upsampling to match mel length, while only ever repeating
        predicted_ids = [resample_np(i, mels.shape[2]) for i in predicted_ids.numpy()]
        # go through predicted_ids and replace 0s with the nearest non-zero
        for i in range(len(predicted_ids)):
            # e.g. for a sequence [0, 0, 1, 0, 0, 2, 0, 0, 0, 3, 0, 0] we get [1, 1, 1, 1, 2, 2, 2, 2, 3, 3, 3]
            predicted_ids[i] = fill_sequence(predicted_ids[i])

        # unbatch mels using mask
        mels = [mels[i].T[mels_mask[i]] for i in range(len(mels))]
        # save mels as images (compressed with maxiumum quality)
        # save phones as ushorts

        batched_speaker = []
        batched_speaker_global = []

        for i in range(len(batch)):
            utt_hash = hashes[i]
            save_dir = book_dirs[i]
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
            # detect silence, which is energy < 0.01 over at least 5 frames
            p_energy = energy.cpu().numpy()
            p_energy = signal.convolve(energy, np.ones(5) / 5, mode="same")
            tres = 0.01
            p_energy = (p_energy < tres)
            # we use different categories of silence (short < 25 frames, intermediate < 50 frames, long > 50 frames)
            # replace phone_list items with the detected silence
            silences_indicies_as_tuples = []
            silence_started = False
            for j, p in enumerate(p_energy):
                if p:
                    if not silence_started:
                        silence_started = True
                        if j <= 5:
                            j = 0
                        silence_start = j
                else:
                    if silence_started:
                        silence_started = False
                        if j >= len(p_energy) - 5:
                            j = len(p_energy) - 1
                        silence_end = j + 1
                        silences_indicies_as_tuples.append((silence_start, silence_end))
            if silence_started:
                silences_indicies_as_tuples.append((silence_start, len(p_energy)))
            # merge silences that are close together
            merged_silences = []
            for silence_start, silence_end in silences_indicies_as_tuples:
                if len(merged_silences) == 0:
                    merged_silences.append((silence_start, silence_end))
                else:
                    if silence_start - merged_silences[-1][1] < 5:
                        merged_silences[-1] = (
                            merged_silences[-1][0],
                            silence_end,
                        )
                    else:
                        merged_silences.append((silence_start, silence_end))
            for silence_start, silence_end in merged_silences:
                if silence_end - silence_start < 25 and silence_end - silence_start > 5:
                    predicted_ids[i][silence_start:silence_end] = len(self.id2phone) - 1
                elif silence_end - silence_start < 50:
                    predicted_ids[i][silence_start:silence_end] = len(self.id2phone) - 2
                else:
                    predicted_ids[i][silence_start:silence_end] = len(self.id2phone) - 3
            # if an id is repeated, we take the first one and set the rest to 0
            predicted_ids[i] = [
                l if l != predicted_ids[i][k - 1] else 0
                for k, l in enumerate(predicted_ids[i])
            ]
            res = zero_split(np.array(predicted_ids[i]))
            phone_list, phone_lengths = res[0], res[1]

            # phones
            np.save(
                save_dir / f"{utt_hash}_phones.npy", phone_list.astype(np.ushort)
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
