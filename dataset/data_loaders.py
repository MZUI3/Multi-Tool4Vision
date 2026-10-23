# Import Basic Lib
import argparse

import torch
from torch.utils.data import DataLoader

# Import Custom Lib
from dataset.dataset import get_datasets
from dataset.sampler import create_sampler


def create_train_val_loader(opts: argparse.Namespace):
    batch_size = getattr(opts, "batch_size", 64)
    num_workers = getattr(opts, "num_workers", 4)
    train_shuffle = getattr(opts, "train_shuffle", True)
    val_shuffle = getattr(opts, "val_shuffle", False)
    pin_memory = torch.cuda.is_available() and str(getattr(opts, "device", "cpu")).startswith("cuda")

    dataset = get_datasets(opts)
    train_sampler = create_sampler(dataset["train"], opts, shuffle=train_shuffle)
    val_sampler = create_sampler(dataset["validation"], opts, shuffle=val_shuffle, evaluation=True)

    common = dict(
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
        persistent_workers=num_workers > 0,
    )
    train_loader = DataLoader(
        dataset=dataset["train"],
        shuffle=train_sampler is None and train_shuffle,
        sampler=train_sampler,
        # A trailing batch of size 1 breaks BatchNorm in train mode.
        drop_last=True,
        **common,
    )
    val_loader = DataLoader(
        dataset=dataset["validation"],
        shuffle=val_sampler is None and val_shuffle,
        sampler=val_sampler,
        **common,
    )

    return train_loader, val_loader, train_sampler, val_sampler
