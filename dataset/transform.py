# Import Basic Lib
import argparse
from torchvision import transforms

# Import Custom Lib
from dataset.info import get_dataset_info


def transform(opts: argparse.Namespace, **kwargs):
    info = get_dataset_info(getattr(opts, "dataset", "imagenet"))
    size = int(getattr(opts, "resize", None) or info.native_size)
    normalize = transforms.Normalize(info.mean, info.std)

    if info.augment == "cifar":
        # Fixed-size, pre-cropped images: upsample only when the model needs a larger input.
        resize = [transforms.Resize(size)] if size != info.native_size else []
        train_transform = transforms.Compose(resize + [
            transforms.RandomCrop(size, padding=max(1, size // 8)),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize,
        ])
        val_transform = transforms.Compose(resize + [
            transforms.ToTensor(),
            normalize,
        ])
        return train_transform, val_transform

    # Large images of varying size (ImageNet-style).
    train_transform = transforms.Compose([
        transforms.RandomResizedCrop(size),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        normalize,
    ])
    val_transform = transforms.Compose([
        transforms.Resize(int(round(size * 256 / 224))),
        transforms.CenterCrop(size),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, val_transform
