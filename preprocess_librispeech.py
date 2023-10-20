import os
import argparse
import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from speech_datasets import Preprocessor

dataset = load_dataset("librispeech_asr/librispeech_asr.py")

splits = {
    "train_clean_100": "train.clean.100",
    "train_clean_360": "train.clean.360",
    "train_other_500": "train.other.500",
    "dev_clean": "validation.clean",
    "dev_other": "validation.other",
    "test_clean": "test.clean",
    "test_other": "test.other",
}

parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--target_location", type=str, default="data")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=None)
parser.add_argument("--spawn", type=bool, default=False)
args = parser.parse_args()

if args.spawn:
    torch.multiprocessing.set_start_method("spawn")

for split, name in splits.items():
    BATCH_SIZE = args.batch_size
    dataloader = DataLoader(
        dataset[name],
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=Preprocessor(
            target_location=f"{args.target_location}/{split}",
            device=args.device,
            allow_overwrite=True,
        ),
        num_workers=args.num_workers,
    )

    for i, batch in tqdm(
        enumerate(dataloader), desc=f"Preprocessing {split}", total=len(dataloader)
    ):
        pass
