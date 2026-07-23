# Import Basic Lib
import argparse
from torchvision import transforms


def _normalize_dataset_name(dataset_name: str) -> str:
    normalized = str(dataset_name).lower().replace("_", "").replace(" ", "")
    if normalized in ["imagenet"]:
        return "imagenet"
    if normalized in ["cifar10", "cifar-10"]:
        return "cifar-10"
    if normalized in ["cifar100", "cifar-100"]:
        return "cifar-100"
    raise ValueError(f"Unsupported dataset '{dataset_name}'")


def transform(opts: argparse.Namespace, **kwargs):
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    size = int(getattr(opts, "resize", getattr(opts, "size", 224)))
    data = _normalize_dataset_name(getattr(opts, "dataset", getattr(opts, "data", "imagenet")))

    if data == "imagenet":
        train_transform = transforms.Compose([
            transforms.RandomResizedCrop(size),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize,
        ])

        val_transform = transforms.Compose([
            transforms.Resize(size),
            transforms.CenterCrop(size),
            transforms.ToTensor(),
            normalize,
        ])
        return train_transform, val_transform

    if data in ["cifar-10", "cifar-100"]:
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        val_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        return train_transform, val_transform

    raise ValueError(f"Unsupported dataset '{data}'")
