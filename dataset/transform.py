# Import Basic Lib
import argparse

from torchvision import transforms

# Import Custom Lib
from dataset.info import get_dataset_info


def _auto_augment(opts: argparse.Namespace, cifar_style: bool):
    name = getattr(opts, "auto_augment", "none") or "none"
    if name == "rand":
        return [transforms.RandAugment()]
    if name == "trivial":
        return [transforms.TrivialAugmentWide()]
    if name == "autoaugment":
        policy = transforms.AutoAugmentPolicy.CIFAR10 if cifar_style else transforms.AutoAugmentPolicy.IMAGENET
        return [transforms.AutoAugment(policy)]
    return []


def _random_erase(opts: argparse.Namespace):
    p = float(getattr(opts, "random_erase", 0.0) or 0.0)
    return [transforms.RandomErasing(p=p, value="random")] if p > 0 else []


def build_train_transform(opts: argparse.Namespace, size: int):
    """Training transform producing ``size x size`` tensors (used per resolution by the multi-scale sampler)."""
    info = get_dataset_info(getattr(opts, "dataset", "imagenet"))
    size = int(size)
    normalize = transforms.Normalize(info.mean, info.std)

    if info.augment == "cifar":
        # Fixed-size, pre-cropped images: upsample only when the model needs a larger input.
        resize = [transforms.Resize(size)] if size != info.native_size else []
        return transforms.Compose(
            resize
            + [
                transforms.RandomCrop(size, padding=max(1, size // 8)),
                transforms.RandomHorizontalFlip(),
                transforms.RandomRotation(15),
            ]
            + _auto_augment(opts, cifar_style=True)
            + [transforms.ToTensor(), normalize]
            + _random_erase(opts)
        )

    # Large images of varying size (ImageNet-style).
    return transforms.Compose(
        [transforms.RandomResizedCrop(size), transforms.RandomHorizontalFlip()]
        + _auto_augment(opts, cifar_style=False)
        + [transforms.ToTensor(), normalize]
        + _random_erase(opts)
    )


def build_val_transform(opts: argparse.Namespace, size: int):
    info = get_dataset_info(getattr(opts, "dataset", "imagenet"))
    size = int(size)
    normalize = transforms.Normalize(info.mean, info.std)
    if info.augment == "cifar":
        resize = [transforms.Resize(size)] if size != info.native_size else []
        return transforms.Compose(resize + [transforms.ToTensor(), normalize])
    return transforms.Compose(
        [transforms.Resize(int(round(size * 256 / 224))), transforms.CenterCrop(size), transforms.ToTensor(), normalize]
    )


def transform(opts: argparse.Namespace, **kwargs):
    info = get_dataset_info(getattr(opts, "dataset", "imagenet"))
    size = int(getattr(opts, "resize", None) or info.native_size)
    return build_train_transform(opts, size), build_val_transform(opts, size)
