import argparse
import os
import random
import sys

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from options import get_opts
from dataset import create_train_val_loader
from models import create_model
from trainer import train_one_epoch, validate
from utils.distributed import cleanup_ddp, get_ddp_args, get_rank, is_main_process, setup_ddp


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _normalize_dataset_name(dataset_name: str) -> str:
    normalized = str(dataset_name).lower().replace("_", "").replace(" ", "")
    if normalized in ["imagenet"]:
        return "imagenet"
    if normalized in ["cifar10", "cifar-10"]:
        return "cifar-10"
    if normalized in ["cifar100", "cifar-100"]:
        return "cifar-100"
    raise ValueError(f"Unsupported dataset '{dataset_name}'")


def _get_num_classes(dataset_name: str) -> int:
    normalized = _normalize_dataset_name(dataset_name)
    if normalized == "imagenet":
        return 1000
    if normalized == "cifar-10":
        return 10
    if normalized == "cifar-100":
        return 100
    raise ValueError(f"Unsupported dataset '{dataset_name}'")


def _resolve_device(opts: argparse.Namespace) -> torch.device:
    device_name = getattr(opts, "device", "cpu")
    if getattr(opts, "ddp", False) and not str(device_name).startswith("cpu") and torch.cuda.is_available():
        _, _, local_rank = get_ddp_args(opts)
        return torch.device(f"cuda:{local_rank}")

    if torch.cuda.is_available() and str(device_name).startswith("cuda"):
        return torch.device(device_name)

    return torch.device("cpu")


def run_training(opts: argparse.Namespace) -> None:
    ddp_active = setup_ddp(opts)
    rank = get_rank()
    set_seed(opts.seed + rank)
    device = _resolve_device(opts)

    if is_main_process():
        print(f"Using device: {device}")
        print(f"Distributed training active: {ddp_active}")

    train_loader, val_loader, train_sampler, val_sampler = create_train_val_loader(opts)
    num_classes = _get_num_classes(opts.dataset)
    model = create_model(opts, num_classes).to(device)

    if opts.checkpoint_path:
        checkpoint_path = os.path.expanduser(opts.checkpoint_path)
        if os.path.exists(checkpoint_path):
            checkpoint = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(checkpoint.get("model_state", checkpoint))
            if is_main_process():
                print(f"Loaded checkpoint from {checkpoint_path}")

    if opts.mode == "print":
        if is_main_process():
            print(model)
        cleanup_ddp()
        return

    if opts.mode == "summary":
        if is_main_process():
            print(model)
            try:
                from torchinfo import summary

                summary(model, input_size=(1, 3, opts.resize, opts.resize), device=str(device))
            except ImportError:
                print("Install torchinfo to see the summary: pip install torchinfo")
        cleanup_ddp()
        return

    if ddp_active:
        model = DistributedDataParallel(model, device_ids=[device] if device.type == "cuda" else None)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(
        model.parameters(), lr=opts.lr, momentum=opts.momentum, weight_decay=opts.weight_decay
    )

    if is_main_process():
        os.makedirs(opts.save_dir, exist_ok=True)

    best_validation_acc = 0.0

    for epoch in range(1, opts.epochs + 1):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        if hasattr(val_sampler, "set_epoch"):
            val_sampler.set_epoch(epoch)

        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc = validate(model, val_loader, criterion, device)

        if is_main_process():
            print(
                f"Epoch {epoch}/{opts.epochs} "
                f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
                f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
            )

            if val_acc > best_validation_acc:
                best_validation_acc = val_acc
                checkpoint = {
                    "epoch": epoch,
                    "model_state": model.module.state_dict() if hasattr(model, "module") else model.state_dict(),
                    "optimizer_state": optimizer.state_dict(),
                    "opts": vars(opts),
                }
                model_name = str(opts.model).replace(" ", "_")
                dataset_name = _normalize_dataset_name(opts.dataset).replace("-", "")
                output_path = os.path.join(opts.save_dir, f"{model_name}_{dataset_name}_best.pth")
                torch.save(checkpoint, output_path)
                print(f"Saved best checkpoint to {output_path}")

    cleanup_ddp()


if __name__ == "__main__":
    opts = get_opts(sys.stdin)
    run_training(opts)
