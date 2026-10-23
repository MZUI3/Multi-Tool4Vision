"""Single source of truth for dataset metadata (name normalisation, classes, resolution, stats, augmentation).

``augment`` picks the transform pipeline:
- ``"cifar"``    : pad-and-crop + flip + small rotation. For small, already-cropped images (CIFAR, STL10).
- ``"imagenet"`` : RandomResizedCrop / Resize + CenterCrop. For large images of varying size.
"""
import os
import re
from dataclasses import dataclass, replace
from typing import Optional, Tuple

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


@dataclass(frozen=True)
class DatasetInfo:
    name: str
    num_classes: Optional[int]  # None = inferred from the folder layout (imagefolder)
    native_size: int
    mean: Tuple[float, float, float]
    std: Tuple[float, float, float]
    augment: str = "imagenet"
    downloadable: bool = True
    layout: str = ""  # human-readable hint for datasets that must be prepared manually

    @property
    def is_small(self) -> bool:
        return self.native_size <= 64


DATASETS = {
    "cifar10": DatasetInfo("cifar10", 10, 32, (0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010), "cifar"),
    "cifar100": DatasetInfo("cifar100", 100, 32, (0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762), "cifar"),
    "stl10": DatasetInfo("stl10", 10, 96, (0.4467, 0.4398, 0.4066), (0.2603, 0.2566, 0.2713), "cifar"),
    "imagenet": DatasetInfo(
        "imagenet",
        1000,
        224,
        IMAGENET_MEAN,
        IMAGENET_STD,
        downloadable=False,
        layout="ILSVRC tar files + devkit in <data-root>, or extracted <data-root>/train/<wnid>/ and <data-root>/val/<wnid>/",
    ),
    "food101": DatasetInfo("food101", 101, 224, IMAGENET_MEAN, IMAGENET_STD),
    "flowers102": DatasetInfo("flowers102", 102, 224, IMAGENET_MEAN, IMAGENET_STD),
    "oxfordpets": DatasetInfo("oxfordpets", 37, 224, IMAGENET_MEAN, IMAGENET_STD),
    "imagefolder": DatasetInfo(
        "imagefolder",
        None,
        224,
        IMAGENET_MEAN,
        IMAGENET_STD,
        downloadable=False,
        layout="<data-root>/train/<class>/*.jpg and <data-root>/val/<class>/*.jpg",
    ),
}

_ALIASES = {
    "pets": "oxfordpets",
    "oxfordiiitpet": "oxfordpets",
    "oxfordiiitpets": "oxfordpets",
    "folder": "imagefolder",
    "custom": "imagefolder",
}


def _normalize(name: str) -> str:
    key = re.sub(r"[\s_\-]", "", str(name).lower())
    return _ALIASES.get(key, key)


def has_folder_layout(root: str) -> bool:
    return os.path.isdir(os.path.join(root, "train")) and os.path.isdir(os.path.join(root, "val"))


def count_folder_classes(root: str) -> int:
    train_dir = os.path.join(root, "train")
    if not os.path.isdir(train_dir):
        raise FileNotFoundError(
            f"imagefolder dataset expects {train_dir}/<class>/... and {os.path.join(root, 'val')}/<class>/..."
        )
    classes = sorted(entry.name for entry in os.scandir(train_dir) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"No class folders found under {train_dir}")
    return len(classes)


def get_dataset_info(name: str, data_root: Optional[str] = None, num_classes: Optional[int] = None) -> DatasetInfo:
    """Look up a dataset. ``num_classes`` overrides, otherwise ``data_root`` is scanned for folder datasets."""
    key = _normalize(name)
    if key not in DATASETS:
        raise ValueError(f"Unsupported dataset '{name}'. Supported: {', '.join(DATASETS)}")
    info = DATASETS[key]
    if num_classes is not None:
        info = replace(info, num_classes=num_classes)
    elif info.num_classes is None and data_root is not None:
        info = replace(info, num_classes=count_folder_classes(data_root))
    return info
