import torch
from torch import nn

from utils.distributed import is_dist_available_and_initialized


def _reduce_scalar(value: float, device: torch.device) -> float:
    if not is_dist_available_and_initialized():
        return value
    tensor = torch.tensor(value, device=device, dtype=torch.float64)
    torch.distributed.all_reduce(tensor)
    return tensor.item()


def train_one_epoch(model: nn.Module, loader, criterion, optimizer, device: torch.device):
    model.train()
    running_loss = 0.0
    running_corrects = 0.0
    total_samples = 0.0

    for inputs, labels in loader:
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        batch_size = inputs.size(0)
        running_loss += loss.item() * batch_size
        running_corrects += outputs.argmax(dim=1).eq(labels).sum().item()
        total_samples += batch_size

    running_loss = _reduce_scalar(running_loss, device)
    running_corrects = _reduce_scalar(running_corrects, device)
    total_samples = _reduce_scalar(total_samples, device)

    epoch_loss = running_loss / total_samples if total_samples else 0.0
    epoch_acc = running_corrects / total_samples if total_samples else 0.0
    return epoch_loss, epoch_acc


def validate(model: nn.Module, loader, criterion, device: torch.device):
    model.eval()
    running_loss = 0.0
    running_corrects = 0.0
    total_samples = 0.0

    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)

            outputs = model(inputs)
            loss = criterion(outputs, labels)

            batch_size = inputs.size(0)
            running_loss += loss.item() * batch_size
            running_corrects += outputs.argmax(dim=1).eq(labels).sum().item()
            total_samples += batch_size

    running_loss = _reduce_scalar(running_loss, device)
    running_corrects = _reduce_scalar(running_corrects, device)
    total_samples = _reduce_scalar(total_samples, device)

    epoch_loss = running_loss / total_samples if total_samples else 0.0
    epoch_acc = running_corrects / total_samples if total_samples else 0.0
    return epoch_loss, epoch_acc
