"""Single source of truth for dataset metadata (name normalisation, classes, resolution, stats)."""
import re
from dataclasses import dataclass
from typing import Tuple


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    num_classes: int
    native_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]


DATASETS = {
    "cifar10": DatasetInfo("cifar10", 10, 32, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    "cifar100": DatasetInfo("cifar100", 100, 32, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
    "imagenet": DatasetInfo("imagenet", 1000, 224, (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
}


def get_dataset_info(name: str) -> DatasetInfo:
    key = re.sub(r"[\s_\-]", "", str(name).lower())
    if key not in DATASETS:
        raise ValueError(f"Unsupported dataset '{name}'. Supported: {', '.join(DATASETS)}")
    return DATASETS[key]
