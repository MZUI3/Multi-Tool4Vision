import argparse
import os

import torch
import torch.nn as nn

from options import format_opts, get_opts
from dataset import DATASETS, create_train_val_loader, get_dataset_info
from dataset.multiscale import check_variable_input, resolve_scales
from models import CATEGORIES, get_model_spec, list_models, resolve_input_size
from trainer import train_one_epoch, validate
from utils.checkpoint import count_parameters, load_checkpoint, save_checkpoint
from utils.ema import ModelEMA
from utils.hwaccel import AccelConfig, Accelerator, is_main_process
from utils.mixup import build_mixup
from utils.optim import build_optimizer, build_scheduler, sync_scheduler_lr


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

    print("\nDatasets")
    for info in DATASETS.values():
        classes = info.num_classes if info.num_classes is not None else "inferred from folders"
        extra = f" | {info.layout}" if info.layout else " | --download-data supported"
        print(f"  {info.name:12s} {str(classes):>22s} classes | {info.native_size}px | {info.augment} aug{extra}")


def build_model(opts: argparse.Namespace):
    """Resolve the input size for (model, dataset) and build the model. Updates ``opts.resize``."""
    spec = get_model_spec(opts.model)
    data_root = opts.data_root if os.path.isdir(os.path.join(opts.data_root, "train")) else None
    info = get_dataset_info(opts.dataset, data_root, opts.num_classes)
    if info.num_classes is None:
        raise ValueError(
            f"Dataset '{info.name}' infers its classes from {info.layout}; "
            f"prepare that layout under --data-root {opts.data_root} or pass --num-classes"
        )
    img_size, adjusted = resolve_input_size(spec, opts.resize, info.native_size)
    if adjusted:
        log(f"{spec.name} needs a larger input than {info.native_size}px; resizing {info.name} images to {img_size}px")
    opts.model = spec.name
    opts.resize = img_size
    return spec.build(num_classes=info.num_classes, img_size=img_size), info, spec


def multi_scale_sizes(opts: argparse.Namespace, spec, model: nn.Module, device: torch.device):
    """Resolutions for the MobileViT multi-scale sampler (or ``None`` when it is off)."""
    if not opts.multi_scale:
        return None
    sizes = resolve_scales(
        opts.resize,
        opts.ms_min_scale,
        opts.ms_max_scale,
        opts.ms_num_scales,
        divisor=max(spec.input_size_divisor, 8 if opts.resize >= 64 else 4),
        min_size=spec.min_input_size,
        explicit=opts.ms_sizes,
    )
    check_variable_input(model, sizes, device)
    return sizes


def run(opts: argparse.Namespace) -> None:
    if opts.mode == "list":
        print_model_list(opts.category)
        return

    accel = Accelerator(AccelConfig.from_namespace(opts), log=print)
    try:
        _run(opts, accel)
    finally:
        accel.close()


def _run(opts: argparse.Namespace, accel: Accelerator) -> None:
    accel.seed(opts.seed)
    device = accel.device
    opts.device = str(device)
    model, info, spec = build_model(opts)
    model = model.to(device)

    log(f"Hardware: {accel.summary()}")
    log(
        f"Model: {opts.model} ({count_parameters(model) / 1e6:.2f}M params) | "
        f"dataset: {info.name} ({info.num_classes} classes) | input: {opts.resize}px"
    )

    if opts.mode in ("print", "summary"):
        if accel.is_main:
            print(model)
            if opts.mode == "summary":
                try:
                    from torchinfo import summary

                    summary(model, input_size=(1, 3, opts.resize, opts.resize), device=str(device))
                except ImportError:
                    print("Install torchinfo to see the summary: pip install torchinfo")
        return

    optimizer = build_optimizer(model, opts)
    scheduler = build_scheduler(optimizer, opts)
    scaler = accel.grad_scaler()
    ema = ModelEMA(model, opts.ema_decay) if opts.ema else None
    start_epoch, best_acc = 1, 0.0

    if opts.checkpoint_path:
        state = load_checkpoint(
            opts.checkpoint_path,
            model,
            device,
            optimizer if opts.resume else None,
            scheduler if opts.resume else None,
            scaler if opts.resume else None,
            ema=ema,
            prefer_ema_weights=opts.ema and opts.mode == "eval",
        )
        log(f"Loaded checkpoint from {opts.checkpoint_path}" + (" (EMA weights)" if state["ema_loaded"] and opts.mode == "eval" else ""))
        if ema is not None and not state["ema_loaded"]:
            # The EMA was copied from the untrained model above; restart it from the loaded weights.
            ema = ModelEMA(model, opts.ema_decay)
            log("Checkpoint has no EMA weights; the EMA starts from the loaded weights")
        if opts.resume:
            sync_scheduler_lr(scheduler)
            start_epoch, best_acc = state["epoch"] + 1, state["best_acc"]
            log(f"Resuming from epoch {start_epoch} (best val_acc={best_acc:.4f})")

    sizes = multi_scale_sizes(opts, spec, model, device)
    model = accel.prepare(model, total_batch=opts.batch_size)

    train_loader, val_loader, train_sampler, val_sampler = create_train_val_loader(opts, accel, sizes)
    criterion = nn.CrossEntropyLoss(label_smoothing=opts.label_smoothing)
    mixup = build_mixup(opts, info.num_classes)
    eval_model = ema.module if ema is not None else model

    if sizes:
        log(f"Multi-scale sampler: {train_sampler.describe()} ({len(train_loader)} batches/epoch)")
    if mixup is not None:
        log(f"Mixup alpha={opts.mixup} cutmix alpha={opts.cutmix} prob={opts.mix_prob}")
    if ema is not None:
        log(f"EMA decay={opts.ema_decay} (validation and *_best.pth use the EMA weights)")

    if opts.mode == "eval":
        val_loss, val_acc = validate(eval_model, val_loader, criterion, device, accel.amp, accel, opts.limit_batches)
        log(f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}")
        return

    log(format_opts(opts))
    if accel.is_main:
        os.makedirs(opts.save_dir, exist_ok=True)
    prefix = os.path.join(opts.save_dir, f"{opts.model}_{info.name}")

    for epoch in range(start_epoch, opts.epochs + 1):
        if hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)

        lr = optimizer.param_groups[0]["lr"]
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, scaler, opts.grad_clip, accel, mixup, ema, opts.limit_batches
        )
        val_loss, val_acc = validate(eval_model, val_loader, criterion, device, accel.amp, accel, opts.limit_batches)
        scheduler.step()

        log(
            f"Epoch {epoch}/{opts.epochs} lr={lr:.2e} "
            f"train_loss={train_loss:.4f} train_acc={train_acc:.4f} "
            f"val_loss={val_loss:.4f} val_acc={val_acc:.4f}" + (" (ema)" if ema is not None else "")
        )

        if accel.is_main:
            is_best = val_acc > best_acc
            best_acc = max(best_acc, val_acc)
            save_checkpoint(f"{prefix}_last.pth", model, optimizer, scheduler, scaler, epoch, best_acc, opts, ema)
            if is_best:
                save_checkpoint(f"{prefix}_best.pth", model, optimizer, scheduler, scaler, epoch, best_acc, opts, ema)
                print(f"Saved best checkpoint to {prefix}_best.pth", flush=True)

    log(f"Done. best val_acc={best_acc:.4f}")


if __name__ == "__main__":
    run(get_opts())
