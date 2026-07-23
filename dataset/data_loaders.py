# Import Basic Lib
import argparse
from torch.utils.data import DataLoader

# Import Custom Lib
from dataset.dataset import get_datasets
from dataset.sampler import create_sampler


def create_train_val_loader(opts: argparse.Namespace):
    batch_size = getattr(opts, "batch_size", 64)
    num_workers = getattr(opts, "num_workers", 4)
    train_shuffle = getattr(opts, "train_shuffle", True)
    val_shuffle = getattr(opts, "val_shuffle", False)

    dataset = get_datasets(opts)
    train_sampler = create_sampler(dataset["train"], opts, shuffle=train_shuffle)
    val_sampler = create_sampler(dataset["validation"], opts, shuffle=val_shuffle) if getattr(opts, "ddp", False) else None

    train_loader = DataLoader(
        dataset=dataset["train"],
        shuffle=train_sampler is None and train_shuffle,
        sampler=train_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )
    val_loader = DataLoader(
        dataset=dataset["validation"],
        shuffle=val_sampler is None and val_shuffle,
        sampler=val_sampler,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    return train_loader, val_loader, train_sampler, val_sampler
