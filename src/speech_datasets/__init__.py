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

# for hashing
from hashlib import sha256

from simple_hifigan import Synthesiser
from simple_titanet import prepare_mel, TitaNet


def resample(x, vpw=5):
    return np.interp(np.linspace(0, 1, vpw), np.linspace(0, 1, len(x)), x)


def locality_sensitive_hashing(x, n_bits=20):
    torch.random.manual_seed(0)
    hyperplanes = torch.randn(n_bits, x.shape[-1])
    hashes = torch.matmul(x, hyperplanes.T)
    return hashes


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
    def __init__(self, target_location):
        self.use_vocex = True
        self.phone_processor = Wav2Vec2Processor.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        )
        self.phone_model = Wav2Vec2ForCTC.from_pretrained(
            "facebook/wav2vec2-xlsr-53-espeak-cv-ft"
        )
        self.id2phone = self.phone_processor.tokenizer.decoder
        self.id2phone[len(self.id2phone)] = "<sil>"
        self.synthesiser = Synthesiser()
        if not self.use_vocex:
            self.titanet = TitaNet(encoder_output=True, load_pretrained=True)
            self.titanet.eval()
        self.vocex = Vocex.from_pretrained("cdminix/vocex")
        self.target_location = target_location
        # make target location if it doesn't exist
        Path(self.target_location).mkdir(parents=True, exist_ok=True)
        # check if empty
        if len(os.listdir(self.target_location)) != 0:
            raise ValueError(
                "Target location is not empty, please empty it before running"
            )
        self.target_location = Path(self.target_location)
        self.pitch_range = (50, 300)
        self.energy_range = (0, 0.2)
        self.vad_range = (0, 1)
        self.speaker_range = (-200, 200)
        self.length_range = (0, 50)
        self.mel_range = (-11, 2)
        self.n_planes = 40

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
            h.update(b["audio"].encode())
            h = h.hexdigest()
            hashes.append(h)
            # hash the speaker
            h = sha256()
            h.update(b["speaker"].encode())
            h = h.hexdigest()
            h = humanhash.humanize(h).replace("-", "_")
            spk_dir = self.target_location / h
            spk_dir.mkdir(parents=True, exist_ok=True)
            spk_dirs.append(spk_dir)
            # book
            book = Path(b["audio"]).name.split("_")[1]
            book_hash = sha256()
            book_hash.update(book.encode())
            book_hash = book_hash.hexdigest()
            book_dir = spk_dir / book_hash
            book_dir.mkdir(parents=True, exist_ok=True)
            book_dirs.append(book_dir)

            audio, sr = torchaudio.load(b["audio"])
            audio = audio / torch.max(torch.abs(audio))
            if sr != 16000:
                audio16 = torchaudio.transforms.Resample(sr, 16000)(audio)
                titanet_mel = prepare_mel(audio16, sr=16000)
                if not self.use_vocex:
                    with torch.no_grad():
                        titanet_overall, titanet_encoder = self.titanet(titanet_mel)
                        batched_speaker.append(
                            titanet_encoder.squeeze(0).transpose(0, 1)
                        )
                        batched_speaker_global.append(titanet_overall)
            batched_audio.append(audio.squeeze(0))
            batched_audio16.append(audio16.squeeze(0))
        batched_audio = nn.utils.rnn.pad_sequence(
            batched_audio, batch_first=True, padding_value=0
        )
        batched_audio16 = nn.utils.rnn.pad_sequence(
            batched_audio16, batch_first=True, padding_value=0
        )
        if not self.use_vocex:
            batched_speaker = nn.utils.rnn.pad_sequence(
                batched_speaker, batch_first=True, padding_value=0
            )

        input_values = self.phone_processor(
            batched_audio16.tolist(), return_tensors="pt", sampling_rate=16000
        ).input_values

        # retrieve logits
        with torch.no_grad():
            logits = self.phone_model(input_values).logits

        mels, mels_mask = self.synthesiser.wavs_to_mel(batched_audio, sr=sr)
        # fp16
        # mels = mels.half()

        # resample batched_speaker to match mels
        if not self.use_vocex:
            batched_speaker = signal.resample(
                batched_speaker.numpy(), mels.shape[2], axis=1
            )
            batched_speaker = torch.from_numpy(batched_speaker)
            # dimensionality reduction
            batched_speaker = (batched_speaker - batched_speaker.mean()) / (
                batched_speaker.std() + 1e-7
            )
            batched_speaker = locality_sensitive_hashing(batched_speaker, self.n_planes)
            batched_speaker = torch.clamp(
                batched_speaker, self.speaker_range[0], self.speaker_range[1]
            )
            batched_speaker = (batched_speaker - self.speaker_range[0]) / (
                self.speaker_range[1] - self.speaker_range[0]
            )
            batched_speaker = batched_speaker.numpy()
            batched_speaker_global = batched_speaker_global.numpy()

        # take argmax and decode
        predicted_ids = torch.argmax(logits, dim=-1)

        # upsampling to match mel length, while only ever repeating
        predicted_ids = [resample_np(i, mels.shape[2]) for i in predicted_ids.numpy()]
        # if an id is repeated, we take the first one and set the rest to 0
        predicted_ids = [
            [
                i if i != predicted_ids[j][k - 1] else 0
                for k, i in enumerate(predicted_ids[j])
            ]
            for j in range(len(predicted_ids))
        ]
        # if there is a 0 at the beginning, replace it with a sil
        for k, i in enumerate(predicted_ids):
            if i[0] == 0:
                i[0] = len(self.id2phone) - 1
            if i[int(mels_mask[k].sum()) - 1] == 0:
                i[int(mels_mask[k].sum()) - 1] = len(self.id2phone) - 1
        res = [zero_split(np.array(i)) for i in predicted_ids]
        phone_list, phone_lengths = zip(*res)

        # unbatch mels using mask
        mels = [mels[i].T[mels_mask[i]] for i in range(len(mels))]
        # save mels as images (compressed with maxiumum quality)
        # save phones as ushorts

        if self.use_vocex:
            batched_speaker = []
            batched_speaker_global = []

        for i in range(len(batch)):
            utt_hash = hashes[i]
            save_dir = book_dirs[i]
            # phones
            np.save(
                save_dir / f"{utt_hash}_phones.npy", phone_list[i].astype(np.ushort)
            )
            # vocex
            with torch.no_grad():
                if len(mels[i]) <= 512:
                    vocex_results = self.vocex.model(
                        mels[i].unsqueeze(0), inference=True
                    )
                else:
                    mel_chunks = torch.split(mels[i], 512, dim=0)
                    chunk_results = []
                    for chunk in mel_chunks:
                        chunk_results.append(
                            self.vocex.model(chunk.unsqueeze(0), inference=True)
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
            vad = vocex_results["measures"]["voice_activity_binary"][0]
            vad = torch.clamp(vad, self.vad_range[0], self.vad_range[1])
            vad = (vad - self.vad_range[0]) / (self.vad_range[1] - self.vad_range[0])
            dvec = vocex_results["dvector"][0]
            np.save(save_dir / f"{utt_hash}_dvec.npy", dvec.numpy())
            if self.use_vocex:
                dvec_time = vocex_results["dvector_time"][0]
                dvec_time = locality_sensitive_hashing(dvec_time, self.n_planes)
                dvec_time = torch.clamp(
                    dvec_time, self.speaker_range[0], self.speaker_range[1]
                )
                dvec_time = (dvec_time - self.speaker_range[0]) / (
                    self.speaker_range[1] - self.speaker_range[0]
                )
                batched_speaker.append(dvec_time.numpy())
                batched_speaker_global.append(dvec.numpy())
            current_idx = 0
            vals_per_window = 10
            prosody = np.zeros((len(phone_lengths[i]), vals_per_window * 3 + 1))
            speaker = np.zeros((len(phone_lengths[i]), self.n_planes))
            for j, d in enumerate(phone_lengths[i]):
                d = int(d)
                if d == 0:
                    prosody[j, :] = 0
                    continue
                if current_idx + d > len(pitch):
                    continue
                pitch_window = pitch[current_idx : current_idx + d]
                energy_window = energy[current_idx : current_idx + d]
                va_window = vad[current_idx : current_idx + d]
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
            lengths = np.array(phone_lengths[i])
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
