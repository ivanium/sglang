from typing import List, Optional

import torch
from vllm.distributed import initialize_model_parallel as vllm_initialize_model_parallel
from vllm.distributed.parallel_state import (
    GroupCoordinator,
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
    get_world_group,
    init_model_parallel_group,
)

_SP: Optional[GroupCoordinator] = None


def get_sp_group():
    assert _SP is not None, "sequence parallel group is not initialized"
    return _SP


def init_sequence_parallel_group(
    group_ranks: List[List[int]], local_rank: int, backend: str
) -> GroupCoordinator:
    return GroupCoordinator(
        group_ranks=group_ranks,
        local_rank=local_rank,
        torch_distributed_backend=backend,
        use_pynccl=True,
    )


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    sequence_parallel_size: int = 1,
    backend: Optional[str] = None,
) -> None:
    assert torch.distributed.is_initialized()
    world_size: int = torch.distributed.get_world_size()
    backend = backend or torch.distributed.get_backend(get_world_group().device_group)

    num_sequence_parallel_groups: int = world_size // sequence_parallel_size
    global _SP
    assert _SP is None, "sequence parallel group is already initialized"
    group_ranks = []
    for i in range(num_sequence_parallel_groups):
        ranks = list(
            range(i * sequence_parallel_size, (i + 1) * sequence_parallel_size)
        )
        group_ranks.append(ranks)
    _SP = init_model_parallel_group(group_ranks, get_world_group().local_rank, backend)
    vllm_initialize_model_parallel(
        tensor_model_parallel_size, pipeline_model_parallel_size, backend
    )


def sequence_parallel_is_initialized():
    return _SP is not None


def get_sequence_parallel_world_size():
    return get_sp_group().world_size


def get_sequence_parallel_rank():
    return get_sp_group().rank_in_group


def get_actual_tensor_model_parallel_world_size():
    return get_tensor_model_parallel_world_size() // get_sequence_parallel_world_size()


def get_actual_tensor_model_parallel_rank():
    return get_tensor_model_parallel_rank() // get_sequence_parallel_world_size()
