"""Multi-scale sampling from MobileViT (Mehta & Rastegari, 2022, Sec. 3.3 / Appendix).

Instead of training at one fixed resolution, every batch is drawn at a random resolution from a set
(default: five sizes from 0.625x to 1.25x the base resolution, e.g. 160..320 for 256). Because the
memory of a batch grows with the number of pixels, the batch size is scaled by ``(base / size)^2`` so
GPU usage stays constant: smaller images, bigger batches; bigger images, smaller batches. This improves
robustness to input scale and lets the model see more samples per epoch at the same memory budget.

Implementation
--------------
``MultiScaleBatchSampler`` yields batches of ``(index, resolution)`` tuples. ``TransformedDataset`` accepts
such tuples in ``__getitem__`` and applies a transform built for that resolution, so every item in a batch
has the same shape and the default collate works. Under DDP all ranks draw the same resolution sequence
(same seed) and the same number of batches, so gradient all-reduce stays in lockstep.
"""
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import torch
from torch.utils.data import Dataset, Sampler


class TransformedDataset(Dataset):
    """Wraps a raw (PIL) dataset and applies a resolution-dependent transform.

    ``dataset[i]`` uses ``base_size``; ``dataset[(i, size)]`` uses a transform built for ``size``.
    """

    def __init__(self, base: Dataset, make_transform: Callable[[int], Callable], base_size: int):
        self.base = base
        self.make_transform = make_transform
        self.base_size = int(base_size)
        self._transforms: Dict[int, Callable] = {}

    def transform_for(self, size: int) -> Callable:
        size = int(size)
        if size not in self._transforms:
            self._transforms[size] = self.make_transform(size)
        return self._transforms[size]

    def __getitem__(self, item):
        if isinstance(item, (tuple, list)):
            index, size = item
        else:
            index, size = item, self.base_size
        image, target = self.base[int(index)]
        return self.transform_for(size)(image), target

    def __len__(self) -> int:
        return len(self.base)

    def __getattr__(self, name):  # expose `targets`, `classes`, ... of the wrapped dataset
        if name in ("base", "make_transform", "base_size", "_transforms"):
            raise AttributeError(name)
        return getattr(self.base, name)


def resolve_scales(
    base_size: int,
    min_scale: float = 0.625,
    max_scale: float = 1.25,
    num_scales: int = 5,
    divisor: int = 1,
    min_size: int = 1,
    explicit: Optional[Sequence[int]] = None,
) -> List[int]:
    """Resolutions used by the multi-scale sampler, rounded to multiples of ``divisor``."""
    if explicit:
        sizes = [int(s) for s in explicit]
    else:
        if num_scales == 1:
            sizes = [base_size]
        else:
            step = (max_scale - min_scale) / (num_scales - 1)
            sizes = [base_size * (min_scale + i * step) for i in range(num_scales)]
    step_unit = max(1, int(divisor))
    out = []
    for s in sizes:
        s = max(int(min_size), int(round(s / step_unit)) * step_unit)
        if s not in out:
            out.append(s)
    out.sort()
    return out


class MultiScaleBatchSampler(Sampler[List[Tuple[int, int]]]):
    """Batch sampler yielding ``[(index, size), ...]`` with a random size per batch.

    The batch size for resolution ``s`` is ``round(batch_size * base_size^2 / s^2)`` (or constant when
    ``scale_batch`` is False). ``weights`` enables class-balanced sampling with replacement.
    Call ``set_epoch`` each epoch so the shuffling / resolution sequence changes.
    """

    def __init__(
        self,
        num_samples: int,
        batch_size: int,
        base_size: int,
        sizes: Iterable[int],
        num_replicas: int = 1,
        rank: int = 0,
        shuffle: bool = True,
        seed: int = 0,
        weights: Optional[Sequence[float]] = None,
        scale_batch: bool = True,
    ):
        self.num_samples = int(num_samples)
        self.batch_size = int(batch_size)
        self.base_size = int(base_size)
        self.sizes = sorted({int(s) for s in sizes}) or [self.base_size]
        self.num_replicas = max(1, int(num_replicas))
        self.rank = int(rank)
        self.shuffle = shuffle
        self.seed = int(seed)
        self.weights = None if weights is None else torch.as_tensor(list(weights), dtype=torch.double)
        self.scale_batch = scale_batch
        self.epoch = 0
        self._plan_cache: Optional[Tuple[int, List[List[Tuple[int, int]]]]] = None
        self.batch_sizes = {
            s: (max(1, int(round(self.batch_size * (self.base_size / s) ** 2))) if scale_batch else self.batch_size)
            for s in self.sizes
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def _plan(self) -> List[List[Tuple[int, int]]]:
        if self._plan_cache is not None and self._plan_cache[0] == self.epoch:
            return self._plan_cache[1]
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch)
        if self.weights is not None:
            indices = torch.multinomial(self.weights, self.num_samples, replacement=True, generator=g)
        elif self.shuffle:
            indices = torch.randperm(self.num_samples, generator=g)
        else:
            indices = torch.arange(self.num_samples)
        per_rank = self.num_samples // self.num_replicas
        local = indices[self.rank::self.num_replicas][:per_rank].tolist()

        batches, pos = [], 0
        while True:
            size = self.sizes[int(torch.randint(len(self.sizes), (1,), generator=g))]
            bs = self.batch_sizes[size]
            if pos + bs > per_rank:
                break
            batches.append([(int(i), size) for i in local[pos:pos + bs]])
            pos += bs
        self._plan_cache = (self.epoch, batches)
        return batches

    def __iter__(self):
        return iter(self._plan())

    def __len__(self) -> int:
        return len(self._plan())

    def describe(self) -> str:
        return ", ".join(f"{s}px x{self.batch_sizes[s]}" for s in self.sizes)


@torch.no_grad()
def check_variable_input(model: torch.nn.Module, sizes: Sequence[int], device: torch.device, channels: int = 3) -> None:
    """Fail early if the model cannot process the smallest / largest multi-scale resolution
    (fixed positional embeddings, window partitions, ...)."""
    was_training = model.training
    model.eval()
    try:
        for size in (min(sizes), max(sizes)):
            try:
                model(torch.zeros(2, channels, size, size, device=device))
            except Exception as exc:  # noqa: BLE001 - any failure means the model is resolution-bound
                raise ValueError(
                    f"--multi-scale: the model does not accept a {size}x{size} input ({type(exc).__name__}: {exc}). "
                    "Multi-scale training needs a resolution-agnostic architecture (CNNs, MobileViT, PVT, ...); "
                    "models with fixed positional embeddings (ViT, DeiT, CaiT, MLP-Mixer, CCT, CoAtNet, Swin, MaxViT) "
                    "must be trained at a single resolution."
                ) from exc
    finally:
        model.train(was_training)
