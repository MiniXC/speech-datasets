import os

from datasets import load_dataset
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from speech_datasets import Preprocessor

train_dataset = load_dataset("cdminix/libritts-aligned", split="dev.clean")

BATCH_SIZE = 4
NUM_BATCHES = 256

dataloader = DataLoader(
    train_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    collate_fn=Preprocessor(target_location="data/test"),
)

for i, batch in tqdm(enumerate(dataloader), total=NUM_BATCHES):
    if i >= (NUM_BATCHES - 1):
        break
