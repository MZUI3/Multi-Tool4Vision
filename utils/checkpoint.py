import os
import warnings

import torch
from torch import nn


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if hasattr(model, "module") else model


def save_checkpoint(path: str, model: nn.Module, optimizer, scheduler, scaler, epoch: int, best_acc: float, opts) -> None:
    state = {
        "epoch": epoch,
        "best_acc": best_acc,
        "model_state": unwrap_model(model).state_dict(),
        "optimizer_state": optimizer.state_dict(),
        "scheduler_state": scheduler.state_dict() if scheduler is not None else None,
        "scaler_state": scaler.state_dict() if scaler is not None else None,
        "opts": vars(opts),
    }
    tmp_path = path + ".tmp"
    torch.save(state, tmp_path)
    os.replace(tmp_path, path)  # never leave a half-written checkpoint behind


def load_checkpoint(
    path: str,
    model: nn.Module,
    device: torch.device,
    optimizer=None,
    scheduler=None,
    scaler=None,
) -> dict:
    """Load weights (and optionally training state). Accepts both full checkpoints and bare state dicts.

    Returns ``{"epoch": last finished epoch, "best_acc": best validation accuracy}``.
    """
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    state_dict = checkpoint.get("model_state", checkpoint)
    state_dict = {k[len("module."):] if k.startswith("module.") else k: v for k, v in state_dict.items()}
    unwrap_model(model).load_state_dict(state_dict)

    if optimizer is not None and checkpoint.get("optimizer_state"):
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        except ValueError as exc:  # e.g. checkpoints written before decay/no-decay param groups existed
            warnings.warn(f"Optimizer state not restored ({exc}); continuing with a fresh optimizer.")
    if scheduler is not None and checkpoint.get("scheduler_state"):
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and checkpoint.get("scaler_state"):
        scaler.load_state_dict(checkpoint["scaler_state"])

    return {"epoch": int(checkpoint.get("epoch", 0)), "best_acc": float(checkpoint.get("best_acc", 0.0))}


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
