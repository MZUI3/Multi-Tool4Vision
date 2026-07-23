import os

import torch

def is_dist_available_and_initialized() -> bool:
    return torch.distributed.is_available() and torch.distributed.is_initialized()


def get_ddp_args(opts):
    rank = int(os.environ.get("RANK", getattr(opts, "rank", 0)))
    world_size = int(os.environ.get("WORLD_SIZE", getattr(opts, "world_size", 1)))
    local_rank = int(os.environ.get("LOCAL_RANK", getattr(opts, "local_rank", 0)))
    return rank, world_size, local_rank


def setup_ddp(opts) -> bool:
    if not getattr(opts, "ddp", False):
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


def reduce_tensor(value: torch.Tensor, op: torch.distributed.ReduceOp = torch.distributed.ReduceOp.SUM) -> torch.Tensor:
    if not is_dist_available_and_initialized():
        return value
    torch.distributed.all_reduce(value, op=op)
    return value
