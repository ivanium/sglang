"""Radix attention."""

import torch
from flashinfer.cascade import merge_state
from torch import nn
from vllm.distributed import (
    get_tensor_model_parallel_rank,
    get_tp_group,
)

from sglang.global_config import global_config
from sglang.srt.layers.extend_attention import extend_attention_fwd

from sglang.srt.layers.parallel_utils.parallel_state import (
    get_sequence_parallel_next_rank,
    get_sequence_parallel_prev_rank,
    get_sequence_parallel_world_size,
)
from sglang.srt.layers.token_attention import token_attention_fwd
from sglang.srt.managers.controller.model_runner import ForwardMode, InputMetadata
from sglang.srt.server import global_server_args_dict


class RadixAttention(nn.Module):
    def __init__(
        self,
        num_heads: int,
        head_dim: int,
        scaling: float,
        num_kv_heads: int,
        layer_id: int,
        logit_cap: int = -1,
    ):
        super().__init__()
        self.tp_q_head_num = num_heads
        self.tp_k_head_num = num_kv_heads
        self.tp_v_head_num = num_kv_heads
        self.head_dim = head_dim
        self.scaling = scaling
        self.layer_id = layer_id

        if not global_server_args_dict.get("disable_flashinfer", False):
            self.extend_forward = self.extend_forward_flashinfer
            self.decode_forward = self.decode_forward_flashinfer
        else:
            self.extend_forward = self.extend_forward_triton
            self.decode_forward = self.decode_forward_triton

        self.logit_cap = logit_cap if logit_cap is not None and logit_cap > 0 else 0

    def extend_forward_triton(self, q, k, v, input_metadata: InputMetadata):
        o = torch.empty_like(q)
        self.store_kv_cache(k, v, input_metadata)
        extend_attention_fwd(
            q.view(-1, self.tp_q_head_num, self.head_dim),
            k.contiguous(),
            v.contiguous(),
            o.view(-1, self.tp_q_head_num, self.head_dim),
            input_metadata.token_to_kv_pool.get_key_buffer(self.layer_id),
            input_metadata.token_to_kv_pool.get_value_buffer(self.layer_id),
            input_metadata.req_to_token_pool.req_to_token,
            input_metadata.req_pool_indices,
            input_metadata.triton_start_loc,
            input_metadata.seq_lens,
            input_metadata.triton_prefix_lens,
            input_metadata.extend_start_loc,
            input_metadata.extend_seq_lens,
            input_metadata.triton_max_seq_len,
            input_metadata.triton_max_extend_len,
            sm_scale=self.scaling,
            logit_cap=self.logit_cap,
        )

        return o

    def decode_forward_triton(self, q, k, v, input_metadata: InputMetadata):
        o = torch.empty_like(q)
        self.store_kv_cache(k, v, input_metadata)

        token_attention_fwd(
            q.view(-1, self.tp_q_head_num, self.head_dim),
            input_metadata.token_to_kv_pool.get_key_buffer(self.layer_id),
            input_metadata.token_to_kv_pool.get_value_buffer(self.layer_id),
            o.view(-1, self.tp_q_head_num, self.head_dim),
            input_metadata.req_to_token_pool.req_to_token,
            input_metadata.req_pool_indices,
            input_metadata.triton_start_loc,
            input_metadata.seq_lens,
            input_metadata.triton_max_seq_len,
            input_metadata.total_num_tokens,
            sm_scale=self.scaling,
            logit_cap=self.logit_cap,
        )

        return o

    def extend_forward_flashinfer(self, q, k, v, input_metadata: InputMetadata):
        o1, s1 = input_metadata.flashinfer_prefill_wrapper_ragged.forward_return_lse(
            q.contiguous().view(-1, self.tp_q_head_num, self.head_dim),
            k.contiguous().view(-1, self.tp_k_head_num, self.head_dim),
            v.contiguous().view(-1, self.tp_v_head_num, self.head_dim),
            causal=True,
            sm_scale=self.scaling,
            logits_soft_cap=self.logit_cap,
        )

        if input_metadata.extend_no_prefix:
            o = o1
        else:
            o2, s2 = input_metadata.flashinfer_prefill_wrapper_paged.forward_return_lse(
                q.contiguous().view(-1, self.tp_q_head_num, self.head_dim),
                input_metadata.token_to_kv_pool.kv_data[self.layer_id],
                causal=False,
                sm_scale=self.scaling,
                logits_soft_cap=self.logit_cap,
            )

            o, _ = merge_state(o1, s1, o2, s2)

        self.store_kv_cache(k, v, input_metadata)

        if input_metadata.total_num_tokens >= global_config.layer_sync_threshold:
            torch.cuda.synchronize()

        return o.view(-1, self.tp_q_head_num * self.head_dim)

    def decode_forward_flashinfer(self, q, k, v, input_metadata: InputMetadata):
        self.store_kv_cache(k, v, input_metadata)

        o = input_metadata.flashinfer_decode_wrapper.forward(
            q.contiguous().view(-1, self.tp_q_head_num, self.head_dim),
            input_metadata.token_to_kv_pool.kv_data[self.layer_id],
            sm_scale=self.scaling,
            logits_soft_cap=self.logit_cap,
        )

        return o.view(-1, self.tp_q_head_num * self.head_dim)

    def send_kv(self, k, v, my_rank, to_rank):
        assert k.size() == v.size()
        tp_group = get_tp_group()
        tp_group.send_object(k.size(), to_rank)
        tp_group.send(k, to_rank)
        tp_group.send(v, to_rank)

    def recv_kv(self, from_rank, dtype, kv_size):
        tp_group = get_tp_group()
        kv_size = tp_group.recv_object(from_rank)
        k = tp_group.recv(kv_size, dtype, from_rank)
        v = tp_group.recv(kv_size, dtype, from_rank)
        return k, v

    def seq_parallel_extend_forward_triton(
        self, q, k, v, input_metadata: InputMetadata
    ):
        raise NotImplementedError()

    def seq_parallel_decode_forward_triton(
        self, q, k, v, input_metadata: InputMetadata
    ):
        raise NotImplementedError()

    def seq_parallel_extend_forward_flashinfer(
        self, q, k, v, input_metadata: InputMetadata
    ):
        def append_merge_shard(shard_list, o, s):
            if len(shard_list) == 0:
                shard_list.append((o, s))
            else:
                o_prev, s_prev = shard_list[-1]
                o, s = merge_state(o_prev, s_prev, o, s)
                shard_list[-1] = (o, s)

        rank = get_tensor_model_parallel_rank()
        sp_size = get_sequence_parallel_world_size()
        num_shards = sp_size
        num_iters = sp_size
        qo_len_per_iter = q.size(1) // num_iters
        batch_size = input_metadata.batch_size

        # FIXME: k and v should have been sharded and trimmed (padding tokens) so use them directly.
        local_k = k.contiguous().view(-1, self.tp_k_head_num, self.head_dim)
        local_v = v.contiguous().view(-1, self.tp_v_head_num, self.head_dim)

        owned_pids = [rank]
        owned_shards = [None for _ in range(num_shards)]
        owned_shards[rank] = (local_k, local_v)
        output_shards = [[] for _ in range(num_shards)]

        # For communication
        to_rank = rank  # which SP worker to send my sequence KV shard to.
        from_rank = rank  # which SP worker to receive the sequence KV shard from.
        pid = rank  # start from the worker's own shard
        for _ in range(num_iters):
            to_rank = get_sequence_parallel_next_rank(to_rank)
            from_rank = get_sequence_parallel_prev_rank(from_rank)
            # FIXME: send-recv communication here
            # if rank != to_rank:
            #     print("send", rank, to_rank)
            #     self.send_kv(local_k, local_v, rank, to_rank)
            #     print("send done", rank, to_rank)
            q_shard = q[:, pid * qo_len_per_iter : (pid + 1) * qo_len_per_iter]
            k_shard, v_shard = owned_shards[pid]
            # Ragged attention computation for self attention within the shard
            o, s = input_metadata.flashinfer_prefill_wrapper_ragged.forward_return_lse(
                q_shard.contiguous().view(-1, self.tp_q_head_num, self.head_dim),
                k_shard.contiguous().view(-1, self.tp_k_head_num, self.head_dim),
                v_shard.contiguous().view(-1, self.tp_v_head_num, self.head_dim),
                causal=True,
                sm_scale=self.scaling,
                logits_soft_cap=self.logit_cap,
            )
            append_merge_shard(output_shards[pid], o, s)
            # Paged attention computation for cross shard attention
            # NOTE: below schedule is for load balancing
            for existing_pid in owned_pids:
                if existing_pid == pid:
                    continue
                i, j = (
                    (existing_pid, pid) if existing_pid > pid else (pid, existing_pid)
                )
                q_data = q[:, i * qo_len_per_iter : (i + 1) * qo_len_per_iter]
                # FIXME: should store them into kv cache and use kv cache here.
                kv_data = torch.stack(owned_shards[j], dim=1)
                o, s = (
                    input_metadata.flashinfer_prefill_wrapper_paged.forward_return_lse(
                        q_data.contiguous().view(-1, self.tp_q_head_num, self.head_dim),
                        kv_data.contiguous(),
                        causal=False,
                        sm_scale=self.scaling,
                        logits_soft_cap=self.logit_cap,
                    )
                )
                append_merge_shard(output_shards[i], o, s)

            # # FIXME: send-recv communication here
            # if rank != from_rank:
            #     print("recv", rank, from_rank)
            #     kv_recved = self.recv_kv(from_rank, k.dtype)
            #     owned_pids.append(pid)
            #     owned_shards[pid] = kv_recved
            # FIXME: here we applied a synchronous communication logic. Need to fix this.
            if rank % 2 == 0:
                if rank != to_rank:
                    self.send_kv(local_k, local_v, rank, to_rank)
                if rank != from_rank:
                    kv_recved = self.recv_kv(from_rank, k.dtype, local_k.size())
                    owned_pids.append(from_rank)
                    owned_shards[from_rank] = kv_recved
            else:
                if rank != from_rank:
                    kv_recved = self.recv_kv(from_rank, k.dtype, local_k.size())
                    owned_pids.append(from_rank)
                    owned_shards[from_rank] = kv_recved
                if rank != to_rank:
                    self.send_kv(local_k, local_v, rank, to_rank)
            pid = from_rank

        # Reshape all o tensors so that we can concatenate along the sequence dimension
        # we must have len(shard_list) == 1 here
        os = [
            o.view(batch_size, qo_len_per_iter, self.tp_q_head_num, self.head_dim)
            for shard_list in output_shards
            for o, _ in shard_list
        ]
        o = torch.cat(os, dim=1)

        # FIXME: enable kv cache storage after we supoprt it.
        # self.store_kv_cache(k, v, input_metadata)

        # if input_metadata.total_num_tokens >= global_config.layer_sync_threshold:
        #     torch.cuda.synchronize()

        return o.view(-1, self.tp_q_head_num * self.head_dim)

    def seq_parallel_decode_forward_flashinfer(
        self, q, k, v, input_metadata: InputMetadata
    ):
        # TODO: implementation
        raise NotImplementedError()

    def forward(self, q, k, v, input_metadata: InputMetadata):
        k = k.view(-1, self.tp_k_head_num, self.head_dim)
        v = v.view(-1, self.tp_v_head_num, self.head_dim)

        if input_metadata.forward_mode == ForwardMode.EXTEND:
            return self.extend_forward(q, k, v, input_metadata)
        elif input_metadata.forward_mode == ForwardMode.DECODE:
            return self.decode_forward(q, k, v, input_metadata)

    def store_kv_cache(self, cache_k, cache_v, input_metadata: InputMetadata):
        kv_cache = input_metadata.token_to_kv_pool.kv_data[self.layer_id]
        _store_kv_cache(cache_k, cache_v, kv_cache, input_metadata.out_cache_loc)


try:

    @torch.library.custom_op("mylib::store_kv_cache", mutates_args={"kv_cache"})
    def _store_kv_cache(
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: torch.Tensor,
        cache_loc: torch.Tensor,
    ) -> None:
        kv_cache[cache_loc, 0] = k
        kv_cache[cache_loc, 1] = v

    @_store_kv_cache.register_fake
    def _(k, v, kv_cache, cache_loc):
        pass

except:

    def _store_kv_cache(
        k: torch.Tensor,
        v: torch.Tensor,
        kv_cache: torch.Tensor,
        cache_loc: torch.Tensor,
    ) -> None:
        kv_cache[cache_loc, 0] = k
        kv_cache[cache_loc, 1] = v
