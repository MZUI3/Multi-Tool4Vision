# Import Basic Lib
import argparse


def _get_option(opts: argparse.Namespace, keys, default=None):
    for opt in keys:
        if hasattr(opts, opt):
            return getattr(opts, opt)
    return default


def _normalize_dataset_name(dataset_name: str) -> str:
    normalized = str(dataset_name).lower().replace("_", "").replace(" ", "")
    if normalized in ["imagenet"]:
        return "imagenet"
    if normalized in ["cifar10", "cifar-10"]:
        return "cifar-10"
    if normalized in ["cifar100", "cifar-100"]:
        return "cifar-100"
    raise ValueError(f"Unsupported dataset '{dataset_name}'")


def _normalize_model_name(model_name: str) -> str:
    normalized = str(model_name).lower().replace("_", "").replace(" ", "")
    if normalized.startswith("resnet"):
        return "resnet"
    if normalized.startswith("densenet"):
        return "densenet"
    if normalized.startswith("mobilenet"):
        return "mobilenet"
    if normalized in ["mobilevit", "exmobilevit"]:
        return normalized
    return normalized


# Get datasets
def get_datasets(opts: argparse.Namespace, **kwargs):
    data = _normalize_dataset_name(_get_option(opts, ["dataset", "data"], "imagenet"))
    model = _normalize_model_name(_get_option(opts, ["model", "Model"], "resnet"))

    pairing_dict = {
        "resnet": ["imagenet", "cifar-100", "cifar-10"],
        "mobilenet": ["imagenet", "cifar-100", "cifar-10"],
        "densenet": ["imagenet", "cifar-100", "cifar-10"],
        "mobilevit": ["imagenet", "cifar-100", "cifar-10"],
        "exmobilevit": ["imagenet", "cifar-100", "cifar-10"],
    }

    if data not in pairing_dict.get(model, []):
        raise ValueError(f"Model '{model}' is not compatible with dataset '{data}'")

    from dataset.transform import transform

    train_transform, val_transform = transform(opts)

    if data == "imagenet":
        from torchvision.datasets import ImageNet

        return {
            "train": ImageNet(root=opts.data_root, split="train", transform=train_transform),
            "validation": ImageNet(root=opts.data_root, split="val", transform=val_transform),
        }

    if data == "cifar-10":
        from torchvision.datasets import CIFAR10

        return {
            "train": CIFAR10(
                root=opts.data_root,
                train=True,
                transform=train_transform,
                download=getattr(opts, "download_data", False),
            ),
            "validation": CIFAR10(
                root=opts.data_root,
                train=False,
                transform=val_transform,
                download=getattr(opts, "download_data", False),
            ),
        }

    if data == "cifar-100":
        from torchvision.datasets import CIFAR100

        return {
            "train": CIFAR100(
                root=opts.data_root,
                train=True,
                transform=train_transform,
                download=getattr(opts, "download_data", False),
            ),
            "validation": CIFAR100(
                root=opts.data_root,
                train=False,
                transform=val_transform,
                download=getattr(opts, "download_data", False),
            ),
        }

    raise ValueError(f"Unsupported dataset '{data}'")
