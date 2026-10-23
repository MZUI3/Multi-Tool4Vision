import collections
import warnings
from typing import Optional

import torch
from torch.utils.data import DistributedSampler, Sampler, WeightedRandomSampler

from utils.distributed import get_ddp_args


class EvalDistributedSampler(Sampler):
    """Splits a dataset across ranks without padding, so every sample is evaluated exactly once.

    ``DistributedSampler`` repeats samples to make the shards equal, which inflates validation
    metrics slightly; uneven shards are fine for evaluation because no gradient sync happens.
    """

    def __init__(self, dataset, num_replicas: int, rank: int):
        self.indices = list(range(rank, len(dataset), num_replicas))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self):
        return len(self.indices)


def _get_dataset_targets(dataset):
    for attr in ("targets", "labels", "_labels"):
        targets = getattr(dataset, attr, None)
        if targets is not None:
            break
    else:
        raise ValueError("Weighted sampler requires a dataset with `targets` or `labels` attribute.")
    if isinstance(targets, torch.Tensor):
        targets = targets.tolist()
    return [int(t) for t in targets]


def create_sampler(dataset, opts, shuffle: bool = True, evaluation: bool = False) -> Optional[Sampler]:
    weighted = getattr(opts, "sampler", "default") == "weighted"

    if getattr(opts, "ddp", False):
        rank, world_size, _ = get_ddp_args(opts)
        if evaluation:
            return EvalDistributedSampler(dataset, world_size, rank)
        if weighted:
            warnings.warn("--sampler weighted is ignored under DDP; DistributedSampler is used instead.")
        return DistributedSampler(dataset, num_replicas=world_size, rank=rank, shuffle=shuffle)

    if evaluation or not weighted:
        return None

    targets = _get_dataset_targets(dataset)
    class_counts = collections.Counter(targets)
    sample_weights = [1.0 / class_counts[target] for target in targets]
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)
