import os
import warnings

import torch
from torch import nn

from utils.hwaccel import strip_module_prefix, unwrap_model


def save_checkpoint(path: str, model: nn.Module, optimizer, scheduler, scaler, epoch: int, best_acc: float, opts, ema=None) -> None:
    state = {
        "epoch": epoch,
        "best_acc": best_acc,
        "model_state": unwrap_model(model).state_dict(),
        "ema_state": ema.state_dict() if ema is not None else None,
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
    ema=None,
    prefer_ema_weights: bool = False,
) -> dict:
    """Load weights (and optionally training state). Accepts both full checkpoints and bare state dicts.

    ``ema`` receives the saved EMA state when present. With ``prefer_ema_weights`` the EMA weights (if saved)
    are loaded into ``model`` itself, which is what you want for evaluation / fine-tuning from an EMA run.

    Returns ``{"epoch": last finished epoch, "best_acc": best validation accuracy, "ema_loaded": bool}``.
    """
    path = os.path.expanduser(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    checkpoint = torch.load(path, map_location=device, weights_only=True)
    ema_state = checkpoint.get("ema_state") if isinstance(checkpoint, dict) else None
    state_dict = checkpoint.get("model_state", checkpoint)
    if prefer_ema_weights and ema_state:
        state_dict = ema_state["module"]
    unwrap_model(model).load_state_dict(strip_module_prefix(state_dict))

    if ema is not None and ema_state:
        ema.load_state_dict(ema_state)
    if optimizer is not None and checkpoint.get("optimizer_state"):
        try:
            optimizer.load_state_dict(checkpoint["optimizer_state"])
        except ValueError as exc:  # e.g. checkpoints written before decay/no-decay param groups existed
            warnings.warn(f"Optimizer state not restored ({exc}); continuing with a fresh optimizer.")
    if scheduler is not None and checkpoint.get("scheduler_state"):
        scheduler.load_state_dict(checkpoint["scheduler_state"])
    if scaler is not None and checkpoint.get("scaler_state"):
        scaler.load_state_dict(checkpoint["scaler_state"])

    return {
        "epoch": int(checkpoint.get("epoch", 0)),
        "best_acc": float(checkpoint.get("best_acc", 0.0)),
        "ema_loaded": bool(ema_state) and (ema is not None or prefer_ema_weights),
    }


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad or not trainable_only)
