import os

import torch
from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from speech_datasets import Preprocessor

dataset = load_dataset("../librispeech_asr/librispeech_asr.py")

splits = {
    #"train_clean_100": "train.clean.100",
    "train_clean_360": "train.clean.360",
    "train_other_500": "train.other.500",
    "dev_clean": "validation.clean",
    "dev_other": "validation.other",
    "test_clean": "test.clean",
    "test_other": "test.other",
}

# splits = {
#     "train_clean_100": "train.100",
#     "train_clean_360": "train.360",
#     "dev_clean": "validation",
#     "test_clean": "test",
# }

for split, name in splits.items():
    BATCH_SIZE = 4

    dataloader = DataLoader(
        dataset[name],
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=Preprocessor(
            target_location=f"data/{split}",
            device="cpu",
            allow_overwrite=True,
        ),
        #num_workers=os.cpu_count()//2,
    )

    for i, batch in tqdm(enumerate(dataloader), desc=f"Preprocessing {split}", total=len(dataloader)):
        pass
