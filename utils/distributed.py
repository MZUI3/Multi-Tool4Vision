"""Single-GPU / DataParallel (DP) / DistributedDataParallel (DDP) helpers.

- ``--parallel dp``  : one process, ``nn.DataParallel`` over the GPUs in ``--gpu-ids`` (default: all).
                       ``--batch-size`` is the total batch, split across GPUs.
- ``--parallel ddp`` : one process per GPU, launched with ``torchrun``. ``--batch-size`` is per process.
                       Launching with torchrun enables this mode automatically.
"""
import os
import warnings
from typing import List, Optional

import torch
from torch import nn
from torch.nn.parallel import DataParallel, DistributedDataParallel


def is_dist_available_and_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def launched_by_torchrun() -> bool:
    return int(os.environ.get("WORLD_SIZE", "1")) > 1


def get_ddp_args(opts):
    rank = int(os.environ.get("RANK", getattr(opts, "rank", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", getattr(opts, "world_size", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", getattr(opts, "local_rank", 0)))
    return rank, world_size, local_rank


def parse_gpu_ids(opts) -> List[int]:
    """GPUs used by DataParallel: ``--gpu-ids 0,1`` or every visible device."""
    raw = getattr(opts, "gpu_ids", None)
    if raw:
        ids = [int(x) for x in str(raw).split(",") if x.strip() != ""]
    else:
        ids = list(range(torch.cuda.device_count()))
    for gpu in ids:
        if gpu >= torch.cuda.device_count():
            raise ValueError(f"--gpu-ids {raw}: cuda:{gpu} does not exist ({torch.cuda.device_count()} visible)")
    return ids


def resolve_device(opts) -> torch.device:
    device_name = str(getattr(opts, "device", "cpu"))
    parallel = getattr(opts, "parallel", "none")
    if device_name.startswith("cpu") or not torch.cuda.is_available():
        return torch.device("cpu")

    if parallel == "ddp":
        _, _, local_rank = get_ddp_args(opts)
        return torch.device(f"cuda:{local_rank}")
    if parallel == "dp":
        gpu_ids = parse_gpu_ids(opts)
        return torch.device(f"cuda:{gpu_ids[0]}") if gpu_ids else torch.device("cuda")
    return torch.device(device_name)


def setup_ddp(opts) -> bool:
    if getattr(opts, "parallel", "none") != "ddp":
        return False

    rank, world_size, local_rank = get_ddp_args(opts)
    backend = opts.backend
    if backend is None:
        backend = "nccl" if torch.cuda.is_available() else "gloo"

    os.environ.setdefault("MASTER_ADDR", opts.master_addr)
    os.environ.setdefault("MASTER_PORT", str(opts.master_port))

    torch.distributed.init_process_group(
        backend=backend,
        rank=rank,
        world_size=world_size,
    )

    if torch.cuda.is_available():
        torch.cuda.set_device(local_rank)

    return True


def cleanup_ddp():
    if is_dist_available_and_initialized():
        torch.distributed.destroy_process_group()


def get_world_size() -> int:
    if not is_dist_available_and_initialized():
        return 1
    return torch.distributed.get_world_size()


def get_rank() -> int:
    if not is_dist_available_and_initialized():
        return 0
    return torch.distributed.get_rank()


def is_main_process() -> bool:
    return get_rank() == 0


def wrap_parallel(model: nn.Module, opts, device: torch.device, log=print) -> nn.Module:
    """Wrap ``model`` (already on ``device``) for the requested parallel mode."""
    parallel = getattr(opts, "parallel", "none")

    if parallel == "ddp":
        if getattr(opts, "sync_bn", False):
            if device.type == "cuda":
                model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
            else:
                log("--sync-bn needs CUDA; keeping regular BatchNorm")
        return DistributedDataParallel(model, device_ids=[device.index] if device.type == "cuda" else None)

    if parallel == "dp":
        if getattr(opts, "sync_bn", False):
            log("--sync-bn only applies to DDP; ignored")
        gpu_ids = parse_gpu_ids(opts) if device.type == "cuda" else []
        if len(gpu_ids) < 2:
            log(f"--parallel dp needs at least 2 CUDA devices ({len(gpu_ids)} available); running on {device}")
            return model
        log(f"DataParallel over cuda:{gpu_ids} (total batch {opts.batch_size}, {opts.batch_size // len(gpu_ids)} per GPU)")
        return DataParallel(model, device_ids=gpu_ids, output_device=gpu_ids[0])

    return model


def reduce_tensor(value: torch.Tensor, op: Optional["torch.distributed.ReduceOp"] = None) -> torch.Tensor:
    if not is_dist_available_and_initialized():
        return value
    torch.distributed.all_reduce(value, op=op or torch.distributed.ReduceOp.SUM)
    return value
