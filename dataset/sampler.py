import collections
import warnings
from typing import List, Optional

import torch
from torch.utils.data import DistributedSampler, Sampler, WeightedRandomSampler

from dataset.multiscale import MultiScaleBatchSampler


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


def class_balanced_weights(dataset) -> List[float]:
    targets = _get_dataset_targets(dataset)
    class_counts = collections.Counter(targets)
    return [1.0 / class_counts[target] for target in targets]


def create_sampler(dataset, opts, accel=None, shuffle: bool = True, evaluation: bool = False) -> Optional[Sampler]:
    weighted = getattr(opts, "sampler", "default") == "weighted"
    distributed = accel is not None and accel.is_distributed

    if distributed:
        if evaluation:
            return EvalDistributedSampler(dataset, accel.world_size, accel.rank)
        if weighted:
            warnings.warn("--sampler weighted is ignored under DDP; DistributedSampler is used instead.")
        return DistributedSampler(dataset, num_replicas=accel.world_size, rank=accel.rank, shuffle=shuffle)

    if evaluation or not weighted:
        return None

    sample_weights = class_balanced_weights(dataset)
    return WeightedRandomSampler(sample_weights, num_samples=len(sample_weights), replacement=True)


def create_multiscale_batch_sampler(dataset, opts, accel, sizes, shuffle: bool = True) -> MultiScaleBatchSampler:
    weighted = getattr(opts, "sampler", "default") == "weighted"
    num_replicas, rank = (accel.world_size, accel.rank) if accel is not None and accel.is_distributed else (1, 0)
    return MultiScaleBatchSampler(
        num_samples=len(dataset),
        batch_size=opts.batch_size,
        base_size=opts.resize,
        sizes=sizes,
        num_replicas=num_replicas,
        rank=rank,
        shuffle=shuffle,
        seed=opts.seed,
        weights=class_balanced_weights(dataset) if weighted else None,
        scale_batch=not getattr(opts, "ms_fixed_batch", False),
    )
