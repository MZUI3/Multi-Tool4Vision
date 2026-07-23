import collections
import os
from typing import Optional

import torch
from torch.utils.data import DistributedSampler, Sampler, WeightedRandomSampler


def _resolve_ddp_args(opts):
    rank = int(os.environ.get("RANK", getattr(opts, "rank", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", getattr(opts, "world_size", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", getattr(opts, "local_rank", 0)))
    return rank, world_size, local_rank


def _get_dataset_targets(dataset):
    targets = getattr(dataset, "targets", None)
    if targets is None:
        targets = getattr(dataset, "labels", None)
    if targets is None:
        raise ValueError("Weighted sampler requires a dataset with `targets` or `labels` attribute.")
    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()
    return list(targets)


def create_sampler(dataset, opts, shuffle: bool = True) -> Optional[Sampler]:
    if getattr(opts, "ddp", False):
        rank, world_size, _ = _resolve_ddp_args(opts)
        return DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)

    if getattr(opts, "sampler", "default") == "weighted":
        targets = _get_dataset_targets(dataset)
        class_counts = collections.Counter(targets)
        sample_weights = [1.0 / class_counts[target] for target in targets]
        return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)

    return None
