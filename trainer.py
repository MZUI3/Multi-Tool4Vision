from contextlib import nullcontext
from typing import Optional

import torch
from torch import nn

from utils.distributed import is_dist_available_and_initialized


def _reduce_metrics(metrics: torch.Tensor) -> torch.Tensor:
    if is_dist_available_and_initialized():
        torch.distributed.all_reduce(metrics)
    return metrics


def _autocast(device: torch.device, enabled: bool):
    return torch.autocast(device_type=device.type, dtype=torch.float16) if enabled else nullcontext()


def _finalize(metrics: torch.Tensor):
    loss_sum, correct, total = _reduce_metrics(metrics).tolist()
    if not total:
        return 0.0, 0.0
    return loss_sum / total, correct / total


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    scaler: Optional[torch.amp.GradScaler] = None,
    grad_clip: Optional[float] = None,
):
    model.train()
    use_amp = scaler is not None and scaler.is_enabled()
    # loss sum, correct predictions, sample count — kept on device to avoid a sync per step
    metrics = torch.zeros(3, device=device, dtype=torch.float64)

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad(set_to_none=True)
        with _autocast(device, use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        if use_amp:
            scaler.scale(loss).backward()
            if grad_clip:
                scaler.unscale_(optimizer)
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss.backward()
            if grad_clip:
                nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

        batch_size = inputs.size(0)
        metrics[0] += loss.detach() * batch_size
        metrics[1] += outputs.argmax(dim=1).eq(labels).sum()
        metrics[2] += batch_size

    epoch_loss, epoch_acc = _finalize(metrics)
    if epoch_loss != epoch_loss or epoch_loss == float("inf"):
        raise RuntimeError("Training loss diverged (NaN/inf). Try a lower --lr, --warmup-epochs or --grad-clip.")
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(model: nn.Module, loader, criterion, device: torch.device, amp: bool = False):
    model.eval()
    metrics = torch.zeros(3, device=device, dtype=torch.float64)

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with _autocast(device, amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        batch_size = inputs.size(0)
        metrics[0] += loss.float() * batch_size
        metrics[1] += outputs.argmax(dim=1).eq(labels).sum()
        metrics[2] += batch_size

    return _finalize(metrics)
