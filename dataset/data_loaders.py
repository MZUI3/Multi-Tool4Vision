# Import Basic Lib
import argparse

from torch.utils.data import DataLoader

# Import Custom Lib
from dataset.dataset import get_datasets
from dataset.sampler import create_multiscale_batch_sampler, create_sampler


def create_train_val_loader(opts: argparse.Namespace, accel=None, multi_scale_sizes=None):
    """Returns ``(train_loader, val_loader, train_sampler, val_sampler)``.

    ``train_sampler`` is whatever object owns ``set_epoch`` (a ``DistributedSampler`` or the multi-scale
    batch sampler) or ``None``.
    """
    batch_size = getattr(opts, "batch_size", 64)
    num_workers = getattr(opts, "num_workers", 4)
    train_shuffle = getattr(opts, "train_shuffle", True)
    val_shuffle = getattr(opts, "val_shuffle", False)
    pin_memory = accel is not None and accel.device.type == "cuda"

    dataset = get_datasets(opts)
    common = dict(num_workers=num_workers, pin_memory=pin_memory, persistent_workers=num_workers > 0)

    if multi_scale_sizes:
        train_sampler = create_multiscale_batch_sampler(dataset["train"], opts, accel, multi_scale_sizes, shuffle=train_shuffle)
        train_loader = DataLoader(dataset=dataset["train"], batch_sampler=train_sampler, **common)
    else:
        train_sampler = create_sampler(dataset["train"], opts, accel, shuffle=train_shuffle)
        train_loader = DataLoader(
            dataset=dataset["train"],
            batch_size=batch_size,
            shuffle=train_sampler is None and train_shuffle,
            sampler=train_sampler,
            # A trailing batch of size 1 breaks BatchNorm in train mode.
            drop_last=True,
            **common,
        )

    val_sampler = create_sampler(dataset["validation"], opts, accel, shuffle=val_shuffle, evaluation=True)
    val_loader = DataLoader(
        dataset=dataset["validation"],
        batch_size=batch_size,
        shuffle=val_sampler is None and val_shuffle,
        sampler=val_sampler,
        **common,
    )

    return train_loader, val_loader, train_sampler, val_sampler
