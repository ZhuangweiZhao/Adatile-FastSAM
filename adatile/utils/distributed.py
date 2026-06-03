"""Distributed training utilities.

Supports: DDP (DistributedDataParallel), gradient sync, metric gathering.
"""

from __future__ import annotations

import os
from typing import Optional

import torch
import torch.distributed as dist
from torch import Tensor


def is_distributed() -> bool:
    """Check if running in distributed mode."""
    return dist.is_available() and dist.is_initialized()


def get_rank() -> int:
    """Get current process rank, or 0 if not distributed."""
    return dist.get_rank() if is_distributed() else 0


def get_world_size() -> int:
    """Get total number of processes, or 1 if not distributed."""
    return dist.get_world_size() if is_distributed() else 1


def init_distributed(
    backend: str = "nccl",
    init_method: str = "env://",
    local_rank: Optional[int] = None,
) -> tuple[int, int]:
    """Initialize distributed training.

    Args:
        backend: Communication backend ("nccl", "gloo", "mpi").
        init_method: URL/file for rendezvous.
        local_rank: GPU local rank.

    Returns:
        (rank, world_size) tuple.
    """
    if local_rank is not None:
        os.environ.setdefault("LOCAL_RANK", str(local_rank))

    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        rank = int(os.environ["RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
    else:
        rank = 0
        world_size = 1

    if not dist.is_initialized() and world_size > 1:
        dist.init_process_group(
            backend=backend,
            init_method=init_method,
            world_size=world_size,
            rank=rank,
        )
        torch.cuda.set_device(local_rank or rank % torch.cuda.device_count())

    return rank, world_size


def reduce_tensor(tensor: Tensor, average: bool = True) -> Tensor:
    """Reduce a tensor across all processes.

    Args:
        tensor: Input tensor.
        average: If True, average; if False, sum.

    Returns:
        Reduced tensor.
    """
    if not is_distributed():
        return tensor

    rt = tensor.clone().detach()
    dist.all_reduce(rt, op=dist.ReduceOp.SUM)
    if average:
        rt /= get_world_size()
    return rt


def gather_tensor(tensor: Tensor, dst: int = 0) -> Optional[Tensor]:
    """Gather tensors from all processes to root.

    Args:
        tensor: Input tensor.
        dst: Destination rank.

    Returns:
        Gathered tensor (None on non-dst ranks).
    """
    if not is_distributed():
        return tensor

    world_size = get_world_size()
    gathered = [torch.zeros_like(tensor) for _ in range(world_size)]
    dist.all_gather(gathered, tensor)
    return torch.cat(gathered, dim=0)


def synchronize() -> None:
    """Barrier synchronization across all processes."""
    if is_distributed():
        dist.barrier()
