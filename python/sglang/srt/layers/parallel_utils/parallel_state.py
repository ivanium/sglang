from typing import Optional

from vllm.distributed import initialize_model_parallel as vllm_initialize_model_parallel
from vllm.distributed.parallel_state import (
    get_tensor_model_parallel_rank,
    get_tensor_model_parallel_world_size,
)

_SEQUENCE_PARALLEL_SIZE = None


def initialize_model_parallel(
    tensor_model_parallel_size: int = 1,
    pipeline_model_parallel_size: int = 1,
    sequence_parallel_size: int = 1,
) -> None:
    global _SEQUENCE_PARALLEL_SIZE
    assert _SEQUENCE_PARALLEL_SIZE is None
    _SEQUENCE_PARALLEL_SIZE = sequence_parallel_size
    vllm_initialize_model_parallel(
        tensor_model_parallel_size, pipeline_model_parallel_size
    )


def sequence_parallel_is_initialized():
    return _SEQUENCE_PARALLEL_SIZE is not None


def get_sequence_parallel_world_size():
    assert _SEQUENCE_PARALLEL_SIZE is not None
    return _SEQUENCE_PARALLEL_SIZE


def get_sequence_parallel_local_rank(rank: Optional[int] = None):
    assert _SEQUENCE_PARALLEL_SIZE is not None
    if rank is None:
        rank = get_tensor_model_parallel_rank()
    local_rank = rank % _SEQUENCE_PARALLEL_SIZE
    return local_rank


def get_sequence_parallel_global_rank():
    return get_tensor_model_parallel_rank()


def get_sequence_parallel_first_rank(rank: Optional[int] = None):
    assert _SEQUENCE_PARALLEL_SIZE is not None
    if rank is None:
        rank = get_tensor_model_parallel_rank()
    first_rank = rank // _SEQUENCE_PARALLEL_SIZE * _SEQUENCE_PARALLEL_SIZE
    return first_rank


def get_sequence_parallel_last_rank(rank: Optional[int] = None):
    assert _SEQUENCE_PARALLEL_SIZE is not None
    if rank is None:
        rank = get_tensor_model_parallel_rank()
    last_rank = (
        rank // _SEQUENCE_PARALLEL_SIZE * _SEQUENCE_PARALLEL_SIZE
        + _SEQUENCE_PARALLEL_SIZE
        - 1
    )
    return last_rank


def get_sequence_parallel_next_rank(rank: Optional[int] = None):
    assert _SEQUENCE_PARALLEL_SIZE is not None
    if rank is None:
        rank = get_tensor_model_parallel_rank()
    first_rank = get_sequence_parallel_first_rank(rank)
    next_rank = first_rank + (rank + 1) % _SEQUENCE_PARALLEL_SIZE
    return next_rank


def get_sequence_parallel_prev_rank(rank: Optional[int] = None):
    assert _SEQUENCE_PARALLEL_SIZE is not None
    if rank is None:
        rank = get_tensor_model_parallel_rank()
    first_rank = get_sequence_parallel_first_rank(rank)
    prev_rank = first_rank + (rank - 1) % _SEQUENCE_PARALLEL_SIZE
    return prev_rank


def get_actual_tensor_model_parallel_world_size():
    return get_tensor_model_parallel_world_size() // get_sequence_parallel_world_size()


def get_actual_tensor_model_parallel_rank():
    return get_tensor_model_parallel_rank() // get_sequence_parallel_world_size()
