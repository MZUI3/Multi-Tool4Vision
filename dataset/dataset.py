# Import Basic Lib
import argparse
import os

from torchvision import datasets as tvd

# Import Custom Lib
from dataset.info import count_folder_classes, get_dataset_info, has_folder_layout
from dataset.transform import transform


def _image_folder(root, train_transform, val_transform, expected_classes=None):
    train = tvd.ImageFolder(os.path.join(root, "train"), transform=train_transform)
    val = tvd.ImageFolder(os.path.join(root, "val"), transform=val_transform)
    if train.classes != val.classes:
        raise ValueError(
            f"train/ and val/ under {root} list different classes "
            f"({len(train.classes)} vs {len(val.classes)}); every class folder must exist in both splits"
        )
    if expected_classes is not None and len(train.classes) != expected_classes:
        raise ValueError(f"{root} has {len(train.classes)} classes, expected {expected_classes}")
    return {"train": train, "validation": val}


# Get datasets
def get_datasets(opts: argparse.Namespace, **kwargs):
    root = opts.data_root
    info = get_dataset_info(getattr(opts, "dataset", "imagenet"), root, getattr(opts, "num_classes", None))
    train_transform, val_transform = transform(opts)
    download = getattr(opts, "download_data", False)

    if info.name == "imagefolder":
        return _image_folder(root, train_transform, val_transform, info.num_classes)

    if info.name == "imagenet":
        # Prefer an already extracted train/ val/ tree; fall back to torchvision's tar + devkit loader.
        if has_folder_layout(root):
            return _image_folder(root, train_transform, val_transform, expected_classes=1000)
        return {
            "train": tvd.ImageNet(root=root, split="train", transform=train_transform),
            "validation": tvd.ImageNet(root=root, split="val", transform=val_transform),
        }

    if info.name in ("cifar10", "cifar100"):
        cls = tvd.CIFAR10 if info.name == "cifar10" else tvd.CIFAR100
        return {
            "train": cls(root=root, train=True, transform=train_transform, download=download),
            "validation": cls(root=root, train=False, transform=val_transform, download=download),
        }

    if info.name == "stl10":
        return {
            "train": tvd.STL10(root=root, split="train", transform=train_transform, download=download),
            "validation": tvd.STL10(root=root, split="test", transform=val_transform, download=download),
        }

    if info.name == "food101":
        return {
            "train": tvd.Food101(root=root, split="train", transform=train_transform, download=download),
            "validation": tvd.Food101(root=root, split="test", transform=val_transform, download=download),
        }

    if info.name == "flowers102":
        return {
            "train": tvd.Flowers102(root=root, split="train", transform=train_transform, download=download),
            "validation": tvd.Flowers102(root=root, split="test", transform=val_transform, download=download),
        }

    if info.name == "oxfordpets":
        return {
            "train": tvd.OxfordIIITPet(root=root, split="trainval", transform=train_transform, download=download),
            "validation": tvd.OxfordIIITPet(root=root, split="test", transform=val_transform, download=download),
        }

    raise ValueError(f"No loader implemented for dataset '{info.name}'")
