# Import Basic Lib
import argparse

# Import Custom Lib
from dataset.info import get_dataset_info
from dataset.transform import transform


# Get datasets
def get_datasets(opts: argparse.Namespace, **kwargs):
    info = get_dataset_info(getattr(opts, "dataset", "imagenet"))
    train_transform, val_transform = transform(opts)
    download = getattr(opts, "download_data", False)

    if info.name == "imagenet":
        from torchvision.datasets import ImageNet

        return {
            "train": ImageNet(root=opts.data_root, split="train", transform=train_transform),
            "validation": ImageNet(root=opts.data_root, split="val", transform=val_transform),
        }

    if info.name == "cifar10":
        from torchvision.datasets import CIFAR10 as dataset_cls
    else:
        from torchvision.datasets import CIFAR100 as dataset_cls

    return {
        "train": dataset_cls(root=opts.data_root, train=True, transform=train_transform, download=download),
        "validation": dataset_cls(root=opts.data_root, train=False, transform=val_transform, download=download),
    }
