from typing import Optional

import torch
from torch import nn

from utils.hwaccel import Accelerator

_LOCAL = None


def _accel(accel: Optional[Accelerator]) -> Accelerator:
    """A default single-process accelerator for callers that do not pass one (tests, notebooks)."""
    global _LOCAL
    if accel is not None:
        return accel
    if _LOCAL is None:
        from utils.hwaccel import AccelConfig

        _LOCAL = Accelerator(AccelConfig(device="cpu"))
    return _LOCAL


def _finalize(metrics: torch.Tensor, accel: Accelerator):
    loss_sum, correct, total = accel.reduce(metrics, "sum").tolist()
    if not total:
        return 0.0, 0.0
    return loss_sum / total, correct / total


def train_one_epoch(
    model: nn.Module,
    loader,
    criterion,
    optimizer,
    device: torch.device,
    scaler: Optional["torch.amp.GradScaler"] = None,
    grad_clip: Optional[float] = None,
    accel: Optional[Accelerator] = None,
    mixup=None,
    ema=None,
    limit_batches: Optional[int] = None,
):
    accel = _accel(accel)
    model.train()
    use_amp = scaler is not None and scaler.is_enabled()
    # loss sum, correct predictions, sample count — kept on device to avoid a sync per step
    metrics = torch.zeros(3, device=device, dtype=torch.float64)

    for step, (inputs, labels) in enumerate(loader):
        if limit_batches is not None and step >= limit_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        targets = labels
        if mixup is not None:
            inputs, targets = mixup(inputs, labels)  # soft targets; accuracy is still measured against `labels`

        optimizer.zero_grad(set_to_none=True)
        with accel.autocast(use_amp):
            outputs = model(inputs)
            loss = criterion(outputs, targets)

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
        if ema is not None:
            ema.update(model)

        batch_size = inputs.size(0)
        metrics[0] += loss.detach() * batch_size
        metrics[1] += outputs.argmax(dim=1).eq(labels).sum()
        metrics[2] += batch_size

    epoch_loss, epoch_acc = _finalize(metrics, accel)
    if epoch_loss != epoch_loss or epoch_loss == float("inf"):
        raise RuntimeError("Training loss diverged (NaN/inf). Try a lower --lr, --warmup-epochs or --grad-clip.")
    return epoch_loss, epoch_acc


@torch.no_grad()
def validate(
    model: nn.Module,
    loader,
    criterion,
    device: torch.device,
    amp: bool = False,
    accel: Optional[Accelerator] = None,
    limit_batches: Optional[int] = None,
):
    accel = _accel(accel)
    model.eval()
    metrics = torch.zeros(3, device=device, dtype=torch.float64)

    for step, (inputs, labels) in enumerate(loader):
        if limit_batches is not None and step >= limit_batches:
            break
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        with accel.autocast(amp):
            outputs = model(inputs)
            loss = criterion(outputs, labels)

        batch_size = inputs.size(0)
        metrics[0] += loss.float() * batch_size
        metrics[1] += outputs.argmax(dim=1).eq(labels).sum()
        metrics[2] += batch_size

    return _finalize(metrics, accel)
