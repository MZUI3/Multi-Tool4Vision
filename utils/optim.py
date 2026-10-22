import argparse
import math

import torch
from torch import nn


def _param_groups(model: nn.Module, weight_decay: float):
    """Exclude biases and normalisation / embedding parameters (all 1-D or tokens) from weight decay."""
    decay, no_decay = [], []
    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue
        if param.ndim <= 1 or name.endswith(("cls_token", "dist_token", "pos_embed")):
            no_decay.append(param)
        else:
            decay.append(param)
    return [
        {"params": decay, "weight_decay": weight_decay},
        {"params": no_decay, "weight_decay": 0.0},
    ]


def build_optimizer(model: nn.Module, opts: argparse.Namespace) -> torch.optim.Optimizer:
    groups = _param_groups(model, opts.weight_decay)
    if opts.optimizer == "adamw":
        return torch.optim.AdamW(groups, lr=opts.lr)
    return torch.optim.SGD(groups, lr=opts.lr, momentum=opts.momentum, nesterov=opts.momentum > 0)


def build_scheduler(optimizer: torch.optim.Optimizer, opts: argparse.Namespace):
    """Per-epoch LR multiplier: linear warmup, then cosine / step / constant."""
    warmup = max(0, opts.warmup_epochs)
    total = max(1, opts.epochs)
    min_ratio = opts.min_lr / opts.lr if opts.lr > 0 else 0.0

    def lr_lambda(epoch: int) -> float:
        if epoch < warmup:
            return (epoch + 1) / (warmup + 1)
        if opts.scheduler == "cosine":
            progress = (epoch - warmup) / max(1, total - warmup)
            return min_ratio + (1 - min_ratio) * 0.5 * (1 + math.cos(math.pi * progress))
        if opts.scheduler == "step":
            return opts.gamma ** ((epoch - warmup) // max(1, opts.step_size))
        return 1.0

    return torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)


def sync_scheduler_lr(scheduler: torch.optim.lr_scheduler.LambdaLR) -> None:
    """Re-apply the schedule after resuming, so a changed --epochs takes effect immediately."""
    for group, base_lr, lr_lambda in zip(scheduler.optimizer.param_groups, scheduler.base_lrs, scheduler.lr_lambdas):
        group["lr"] = base_lr * lr_lambda(scheduler.last_epoch)
    scheduler._last_lr = [group["lr"] for group in scheduler.optimizer.param_groups]
