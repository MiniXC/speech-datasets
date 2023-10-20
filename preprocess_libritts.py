import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse

from speech_datasets import Preprocessor

dataset = load_dataset("cdminix/libritts-aligned")

splits = {
    # "train_clean_100": "train.clean.100",
    # "train_clean_360": "train.clean.360",
    # "train_other_500": "train.other.500",
    "dev_clean": "dev.clean",
    # "dev_other": "dev.other",
    # "test_clean": "test.clean",
    # "test_other": "test.other",
}

# argument for device
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")
parser.add_argument("--target_location", type=str, default="data")
parser.add_argument("--batch_size", type=int, default=4)
parser.add_argument("--num_workers", type=int, default=0)
args = parser.parse_args()

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
