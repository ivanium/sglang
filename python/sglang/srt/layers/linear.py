import logging
from typing import Dict, Optional, Tuple

import torch
from torch.nn.parameter import Parameter
from vllm.distributed import divide, get_tensor_model_parallel_world_size
from vllm.model_executor.layers.linear import ColumnParallelLinear, RowParallelLinear
from vllm.model_executor.layers.quantization.base_config import QuantizationConfig

from sglang.srt.layers.parallel_utils.parallel_state import (
    get_actual_tensor_model_parallel_rank,
    get_actual_tensor_model_parallel_world_size,
    get_sequence_parallel_local_rank,
    get_sequence_parallel_world_size,
)

logger = logging.getLogger(__name__)


# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/layers/linear.py#L21
def adjust_marlin_shard(param, shard_size, shard_offset):
    marlin_tile_size = getattr(param, "marlin_tile_size", None)
    if marlin_tile_size is None:
        return shard_size, shard_offset

    return shard_size * marlin_tile_size, shard_offset * marlin_tile_size


# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/layers/linear.py#L29
def adjust_bitsandbytes_shard(
    param: Parameter, kv_offsets: Dict[str, Tuple[int, int]], loaded_shard_id: str
) -> Tuple[int, int]:
    """Adjust the quantization offsets and sizes for BitsAndBytes sharding."""

    total, _ = kv_offsets["total"]
    orig_offset, orig_size = kv_offsets[loaded_shard_id]

    quantized_total = param.data.shape[0]
    quantized_offset = orig_offset * quantized_total // total
    quantized_size = orig_size * quantized_total // total

    return quantized_size, quantized_offset


# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/layers/linear.py#L44
def adjust_scalar_to_fused_array(param, loaded_weight, shard_id):
    """For fused modules (KV) we have an array of length
    N that holds 1 scale for each "logical" matrix. So the param
    is an array of length N. The loaded_weight corresponds to
    one of the shards on disk. Here, we slice the param based on
    the shard_id for loading.
    """
    qkv_idxs = {"k": 0, "v": 1}

    if isinstance(shard_id, str):
        shard_id = qkv_idxs[shard_id]
    elif not isinstance(shard_id, int):
        raise ValueError(f"Unknown Shard Id {shard_id}")

    # AutoFP8 scales do not have a shape
    # compressed-tensors scales do have a shape
    if len(loaded_weight.shape) != 0:
        assert loaded_weight.shape[0] == 1
        loaded_weight = loaded_weight[0]

    return param[shard_id], loaded_weight


class QKVParallelLinear(torch.nn.Module):
    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__()
        # FIXME (yifan): output_size should be head_size * total_num_heads // sp_size.
        # We probably need to manually partition the q_proj in this case since RowPrallelLinear
        # will perform all-gather after the computation.
        # q projection can be naively tensor parallelized
        self.q_proj = RowParallelLinear(
            hidden_size,
            head_size * total_num_heads,
            bias,
            skip_bias_add,
            params_dtype,
            quant_config,
        )
        # kv projection needs both tensor and sequence parallelization
        self.kv_proj = KVSequenceParallelLinear(
            hidden_size,
            head_size,
            total_num_heads,
            total_num_kv_heads,
            bias,
            skip_bias_add,
            params_dtype,
            quant_config,
        )
        self.kv_size = self.kv_proj.num_kv_heads * self.kv_proj.head_size

    def forward(
        self,
        hidden_states: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        q, _ = self.q_proj(hidden_states)
        kv, _ = self.kv_proj(hidden_states)
        k, v = kv.split([self.kv_size, self.kv_size], dim=-1)
        return q, k, v


# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/layers/linear.py#L422
class KVSequenceParallelLinear(ColumnParallelLinear):
    """Linear layers for the attention's KV transformation.

    Linear layers for the linear transformation of the key, and value
    vectors in the attention layer. The weight matrix is concatenated along
    the output dimension. The layer is parallelized along the head dimension.
    When the number of key/value heads is smaller than the number of query
    heads (e.g., multi-query/grouped-query attention), the key/value head may
    be replicated while the query heads are partitioned.

    Args:
        hidden_size: input hidden state size of the transformer.
        head_size: size of each attention head.
        total_num_heads: total number of attention query heads.
        total_num_kv_heads: total number of attention key/value heads. If
                            None, assume total_num_kv_heads = total_num_heads.
        bias: If true, add bias.
        skip_bias_add: This was added to enable performance optimizations where
                       bias can be fused with other element-wise operations. we
                       skip adding bias but instead return it.
        params_dtype: Data type for the parameters.
        linear_method: (Maybe quantized) linear method.
    """

    def __init__(
        self,
        hidden_size: int,
        head_size: int,
        total_num_heads: int,
        total_num_kv_heads: Optional[int] = None,
        bias: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        self.hidden_size = hidden_size
        self.head_size = head_size
        if total_num_kv_heads is None:
            total_num_kv_heads = total_num_heads
        self.total_num_kv_heads = total_num_kv_heads
        # Divide the weight matrix along the last dimension.
        actual_tp_size = get_actual_tensor_model_parallel_world_size()
        if actual_tp_size >= self.total_num_kv_heads:
            self.num_kv_heads = 1
            self.num_kv_head_replicas = divide(actual_tp_size, self.total_num_kv_heads)
        else:
            self.num_kv_heads = divide(self.total_num_kv_heads, actual_tp_size)
            self.num_kv_head_replicas = 1
        input_size = self.hidden_size
        # NOTE (yifan): here we use tp_size to make parent class happy. This is
        # because parent class expects pure tensor parallelism.
        tp_size = get_tensor_model_parallel_world_size()
        output_size = 2 * self.num_kv_heads * tp_size * self.head_size
        self.output_sizes = [
            self.num_kv_heads * self.head_size * tp_size,  # k_proj
            self.num_kv_heads * self.head_size * tp_size,  # v_proj
        ]

        super().__init__(
            input_size=input_size,
            output_size=output_size,
            bias=bias,
            gather_output=False,
            skip_bias_add=skip_bias_add,
            params_dtype=params_dtype,
            quant_config=quant_config,
        )

    def weight_loader(
        self,
        param: Parameter,
        loaded_weight: torch.Tensor,
        loaded_shard_id: Optional[str] = None,
    ):
        param_data = param.data
        output_dim = getattr(param, "output_dim", None)
        # Special case for AQLM codebooks.
        is_metadata = getattr(param, "is_metadata", False)

        # Special case for per-tensor scales in fused case.
        needs_scalar_to_array = getattr(param, "needs_scalar_to_array", False)

        if loaded_shard_id is None:
            # Loaded weight is already fused on disk (qkv/mlp).
            raise NotImplementedError("Fused weight loading is not supported.")
            if output_dim is None:
                assert param_data.shape == loaded_weight.shape
                param_data.copy_(loaded_weight)
                return
            shard_offsets = [
                # (shard_id, shard_offset, shard_size)
                (
                    "k",
                    0,
                    self.total_num_kv_heads * self.head_size,
                ),
                (
                    "v",
                    self.total_num_kv_heads * self.head_size,
                    self.total_num_kv_heads * self.head_size,
                ),
            ]
            packed_dim = getattr(param, "packed_dim", None)
            for shard_id, shard_offset, shard_size in shard_offsets:
                # If quantized, we need to adjust the offset and size to account
                # for the packing.
                if packed_dim == output_dim:
                    shard_size = shard_size // param.pack_factor
                    shard_offset = shard_offset // param.pack_factor
                loaded_weight_shard = loaded_weight.narrow(
                    output_dim, shard_offset, shard_size
                )
                self.weight_loader(param, loaded_weight_shard, shard_id)
            return

        actual_tp_rank = get_actual_tensor_model_parallel_rank()
        assert loaded_shard_id in ["k", "v"]

        # If output dim is defined, use the default loading process.
        if output_dim is not None:
            if loaded_shard_id == "k":
                shard_offset = 0
                shard_size = self.num_kv_heads * self.head_size
            elif loaded_shard_id == "v":
                shard_offset = self.num_kv_heads * self.head_size
                shard_size = self.num_kv_heads * self.head_size
            # Special case for Quantized Weights.
            # If quantized, we need to adjust the offset and size to account
            # for the packing.
            packed_dim = getattr(param, "packed_dim", None)
            if packed_dim == output_dim:
                shard_size = shard_size // param.pack_factor
                shard_offset = shard_offset // param.pack_factor

                # Special case for Marlin.
                shard_size, shard_offset = adjust_marlin_shard(
                    param, shard_size, shard_offset
                )

            use_bitsandbytes = getattr(param, "use_bitsandbytes", False)
            if use_bitsandbytes:
                orig_kv_offsets = {
                    "k": (
                        0,
                        self.num_kv_heads * self.head_size,
                    ),
                    "v": (
                        self.num_kv_heads * self.head_size,
                        self.num_kv_heads * self.head_size,
                    ),
                    "total": (
                        2 * self.num_kv_heads * self.head_size,
                        0,
                    ),
                }
                shard_size, shard_offset = adjust_bitsandbytes_shard(
                    param, orig_kv_offsets, loaded_shard_id
                )

            param_data = param_data.narrow(output_dim, shard_offset, shard_size)
            shard_id = actual_tp_rank // self.num_kv_head_replicas
            start_idx = shard_id * shard_size
            loaded_weight = loaded_weight.narrow(output_dim, start_idx, shard_size)
        # Special case for for AQLM codebooks.
        elif is_metadata:
            # metadata indicates fixed size concatenated along dim 0
            shard_size = loaded_weight.shape[0]
            shard_index = ["k", "v"].index(loaded_shard_id)
            param_data = param_data.narrow(0, shard_index * shard_size, shard_size)
        # Special case for per-tensor scales in fused case.
        elif needs_scalar_to_array:
            param_data, loaded_weight = adjust_scalar_to_fused_array(
                param_data, loaded_weight, loaded_shard_id
            )
        else:
            ignore_warning = getattr(param, "ignore_warning", False)
            if not ignore_warning:
                logger.warning(
                    "Loading a weight without `output_dim` attribute in "
                    "QKVParallelLinear, assume the weight is the same "
                    "for all partitions."
                )
        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


# Adapted from
# https://github.com/vllm-project/vllm/blob/c7f2cf2b7f67bce5842fedfdba508440fe257375/vllm/model_executor/layers/linear.py#L660
class RowSeqParallelLinear(RowParallelLinear):
    """TODO: add doc string."""

    def __init__(
        self,
        input_size: int,
        output_size: int,
        total_num_heads: int,
        num_kv_heads: int,
        head_dim: int,
        bias: bool = True,
        input_is_parallel: bool = True,
        skip_bias_add: bool = False,
        params_dtype: Optional[torch.dtype] = None,
        reduce_results: bool = True,
        quant_config: Optional[QuantizationConfig] = None,
    ):
        super().__init__(
            input_size,
            output_size,
            bias,
            input_is_parallel,
            skip_bias_add,
            params_dtype,
            reduce_results,
            quant_config,
        )
        self.total_num_heads = total_num_heads
        self.num_kv_heads = num_kv_heads
        self.head_dim = head_dim

    def weight_loader(self, param: Parameter, loaded_weight: torch.Tensor):
        actual_tp_rank = get_actual_tensor_model_parallel_rank()
        actual_tp_size = get_actual_tensor_model_parallel_world_size()
        sp_size = get_sequence_parallel_world_size()
        sp_rank = get_sequence_parallel_local_rank()

        input_dim = getattr(param, "input_dim", None)
        param_data = param.data
        if input_dim is not None:
            # Load TP weight shard
            shard_size = param_data.shape[input_dim]
            tp_shard_size = shard_size * sp_size
            start_idx = actual_tp_rank * tp_shard_size
            loaded_weight = loaded_weight.narrow(input_dim, start_idx, tp_shard_size)
            # Load SP weight shard
            tp_num_heads = self.total_num_heads // actual_tp_size
            idxes = _get_sequence_parallel_head_idxes(
                tp_num_heads, self.num_kv_heads, sp_rank, sp_size
            )
            loaded_weight = (
                loaded_weight.contiguous()
                .view(loaded_weight.shape[0], tp_num_heads, self.head_dim)[:, idxes]
                .view(loaded_weight.shape[0], shard_size)
            )

        # Special case for loading scales off disk, which often do not
        # have a shape (such as in the case of AutoFP8).
        if len(loaded_weight.shape) == 0:
            if sp_size > 1:
                raise NotImplementedError("Loading scales off disk is not supported.")
            loaded_weight = loaded_weight.reshape(1)

        assert param_data.shape == loaded_weight.shape
        param_data.copy_(loaded_weight)


def _get_sequence_parallel_head_idxes(total_num_heads, num_kv_heads, sp_rank, sp_size):
    group_size = total_num_heads // num_kv_heads
    shard_num_heads = group_size // sp_size

    idxes = [
        group_size * i + sp_rank * shard_num_heads + j
        for i in range(num_kv_heads)
        for j in range(0, shard_num_heads)
    ]
    return idxes
