import argparse
import os
import random

import torch
import torch.nn as nn
from torch.nn.parallel import DistributedDataParallel

from options import format_opts, get_opts
from dataset import create_train_val_loader, get_dataset_info
from models import CATEGORIES, get_model_spec, list_models, resolve_input_size
from trainer import train_one_epoch, validate
from utils.checkpoint import count_parameters, load_checkpoint, save_checkpoint
from utils.distributed import cleanup_ddp, get_ddp_args, get_rank, is_main_process, setup_ddp
from utils.optim import build_optimizer, build_scheduler, sync_scheduler_lr


def set_seed(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def _resolve_device(opts: argparse.Namespace) -> torch.device:
    device_name = getattr(opts, "device", "cpu")
    if getattr(opts, "ddp", False) and not str(device_name).startswith("cpu") and torch.cuda.is_available():
        _, _, local_rank = get_ddp_args(opts)
        return torch.device(f"cuda:{local_rank}")

    if torch.cuda.is_available() and str(device_name).startswith("cuda"):
        return torch.device(device_name)

    return torch.device("cpu")


def log(message: str) -> None:
    if is_main_process():
        print(message, flush=True)


def print_model_list(category=None) -> None:
    if category is not None and category not in CATEGORIES:
        raise ValueError(f"Unknown category '{category}'. Expected one of {list(CATEGORIES)}")
    for key, title in CATEGORIES.items():
        if category is not None and key != category:
            continue
        specs = list_models(key)
        families = {}
        for spec in specs:
            families.setdefault(spec.family, []).append(spec)
        print(f"\n[{key}] {title} - {len(families)} families / {len(specs)} models")
        for family, members in families.items():
            print(f"  {family:28s} {members[0].paper}")
            print(f"    {', '.join(m.name for m in members)}")


def build_model(opts: argparse.Namespace):
    """Resolve the input size for (model, dataset) and build the model. Updates ``opts.resize``."""
    spec = get_model_spec(opts.model)
    info = get_dataset_info(opts.dataset)
    img_size, adjusted = resolve_input_size(spec, opts.resize, info.native_size)
    if adjusted:
        log(f"{spec.name} needs a larger input than {info.native_size}px; resizing {info.name} images to {img_size}px")
    opts.model = spec.name
    opts.resize = img_size
    return spec.build(num_classes=info.num_classes, img_size=img_size), info


def run(opts: argparse.Namespace) -> None:
    if opts.mode == "list":
        print_model_list(opts.category)
        return

    ddp_active = setup_ddp(opts)
    try:
        _run(opts, ddp_active)
    finally:
        cleanup_ddp()


def _run(opts: argparse.Namespace, ddp_active: bool) -> None:
    set_seed(opts.seed + get_rank())
    device = _resolve_device(opts)
    model, info = build_model(opts)
    model = model.to(device)

    log(f"Using device: {device} | distributed: {ddp_active}")
    log(f"Model: {opts.model} ({count_parameters(model) / 1e6:.2f}M params) | dataset: {info.name} | input: {opts.resize}px")

    if opts.mode in ("print", "summary"):
        if is_main_process():
            print(model)
            if opts.mode == "summary":
                try:
                    from torchinfo import summary

                    summary(model, input_size=(1, 3, opts.resize, opts.resize), device=str(device))
                except ImportError:
                    print("Install torchinfo to see the summary: pip install torchinfo")
        return

    amp = opts.amp and device.type == "cuda"
    if opts.amp and not amp:
        log("--amp is only supported on CUDA; running in full precision")

    optimizer = build_optimizer(model, opts)
    scheduler = build_scheduler(optimizer, opts)
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    start_epoch, best_acc = 1, 0.0

    if opts.checkpoint_path:
        state = load_checkpoint(
            opts.checkpoint_path,
            model,
            device,
            optimizer if opts.resume else None,
            scheduler if opts.resume else None,
            scaler if opts.resume else None,
        )
        log(f"Loaded checkpoint from {opts.checkpoint_path}")
        if opts.resume:
            sync_scheduler_lr(scheduler)
            start_epoch, best_acc = state["epoch"] + 1, state["best_acc"]
            log(f"Resuming from epoch {start_epoch} (best val_acc={best_acc:.4f})")

    if ddp_active:
        if opts.sync_bn and device.type == "cuda":
            model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
        elif opts.sync_bn:
            log("--sync-bn needs CUDA; keeping regular BatchNorm")
        model = DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)

    train_loader, val_loader, train_sampler, val_sampler = create_train_val_loader(opts)
    criterion = nn.CrossEntropyLoss(label_smoothing=opts.label_smoothing)

    if opts.mode == "eval":
        val_loss, val_acc = validate(model, val_loader, criterion, device, amp)
        log(f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        return

    log(format_opts(opts))
    if is_main_process():
        os.makedirs(opts.save_dir, exist_ok=True)
    prefix = os.path.join(opts.save_dir, f"{opts.model}_{info.name}")

    for epoch in range(start_epoch, opts.epochs + 1):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        if hasattr(val_sampler, "set_epoch"):
            val_sampler.set_epoch(epoch)

        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(model, train_loader, criterion, optimizer, device, scaler, opts.grad_clip)
        val_loss, val_acc = validate(model, val_loader, criterion, device, amp)
        scheduler.step()

        log(
            f"Epoch {epoch}/{opts.epochs} lr={lr:.2e} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
        )

        if is_main_process():
            is_best = val_acc > best_acc
            best_acc = max(best_acc, val_acc)
            save_checkpoint(f"{prefix}_last.pth", model, optimizer, scheduler, scaler, epoch, best_acc, opts)
            if is_best:
                save_checkpoint(f"{prefix}_best.pth", model, optimizer, scheduler, scaler, epoch, best_acc, opts)
                print(f"Saved best checkpoint to {prefix}_best.pth", flush=True)

    log(f"Done. best val_acc={best_acc:.4f}")


if __name__ == "__main__":
    run(get_opts())
