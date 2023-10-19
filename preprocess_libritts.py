import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm
import argparse
import numpy as np
import random

from speech_datasets import Preprocessor

dataset = load_dataset("cdminix/libritts-aligned")


def set_seed(seed):
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)


set_seed(42)

splits = {
    "train_clean_100": "train.clean.100",
    "train_clean_360": "train.clean.360",
    "train_other_500": "train.other.500",
    "dev_clean": "dev.clean",
    "dev_other": "dev.other",
    "test_clean": "test.clean",
    "test_other": "test.other",
}

# argument for device
parser = argparse.ArgumentParser()
parser.add_argument("--device", type=str, default="cuda:0")
args = parser.parse_args()

for split, name in splits.items():
    BATCH_SIZE = 4
    # set seed
    dataloader = DataLoader(
        dataset[name],
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=Preprocessor(
            target_location=f"data/{split}",
            device=args.device,
            allow_overwrite=True,
        ),
        # num_workers=os.cpu_count()//2,
    )

    for i, batch in tqdm(
        enumerate(dataloader), desc=f"Preprocessing {split}", total=len(dataloader)
    ):
        pass
