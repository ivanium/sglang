"""Radix attention."""

import torch
from flashinfer.cascade import merge_state
from torch import nn
from torch.distributed import P2POp, batch_isend_irecv, irecv, isend
from vllm.distributed import get_tensor_model_parallel_rank, get_tp_group

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

    def launch_sp_comm_ops(self, kv_to_recv, kv_to_send, from_rank, my_rank, to_rank):
        def _send(handles, group):
            if my_rank != to_rank:
                for t in kv_to_send:
                    handles.append(P2POp(op=isend, tensor=t, peer=to_rank, group=group))

        def _recv(handles, group):
            if my_rank != from_rank:
                for t in kv_to_recv:
                    handles.append(
                        P2POp(op=irecv, tensor=t, peer=from_rank, group=group)
                    )

        handles = []
        reqs = []
        tp_group = get_tp_group().device_group
        # Interleaving workers for send and recv to avoid deadlock
        if my_rank % 2 == 0:
            _send(handles, tp_group)
            _recv(handles, tp_group)
        else:
            _recv(handles, tp_group)
            _send(handles, tp_group)
        if handles:
            reqs = batch_isend_irecv(handles)
        return reqs

    def wait_sp_comm_ops(self, reqs):
        for req in reqs:
            req.wait()

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
        """Here we adopted a unique parallelization strategy.
        For each SP worker, we have
            q tensor: [batch_size, seq_len, q_head_num // SP_SIZE, head_dim]
            k tensor: [batch_size, seq_len // SP_SIZE, k_head_num, head_dim]
            v tensor: [batch_size, seq_len // SP_SIZE, v_head_num, head_dim]
        """

        def get_sp_seq_range(seq_len, sp_rank, sp_size):
            sp_seq_len = seq_len // sp_size + ((seq_len % sp_size) > sp_rank)
            stt = sp_rank * (seq_len // sp_size) + min(sp_rank, seq_len % sp_size)
            end = stt + sp_seq_len
            return stt, end

        def get_k_shard_shape(batch_size, seq_len, sp_rank, sp_size):
            sp_seq_len = seq_len // sp_size + ((seq_len % sp_size) > sp_rank)
            return (batch_size * sp_seq_len, self.tp_k_head_num, self.head_dim)

        def get_v_shard_shape(batch_size, seq_len, sp_rank, sp_size):
            sp_seq_len = seq_len // sp_size + ((seq_len % sp_size) > sp_rank)
            return (batch_size * sp_seq_len, self.tp_v_head_num, self.head_dim)

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
        seq_len = q.size(1)
        batch_size = input_metadata.batch_size

        # FIXME: k and v should have been sharded and trimmed (padding tokens) so use them directly.
        local_k = k.contiguous().view(-1, self.tp_k_head_num, self.head_dim)
        local_v = v.contiguous().view(-1, self.tp_v_head_num, self.head_dim)

        owned_sids = [rank]
        owned_shards = [None for _ in range(num_shards)]
        owned_shards[rank] = (local_k, local_v)
        output_shards = [[] for _ in range(num_shards)]

        # For communication
        to_rank = rank  # which SP worker to send my sequence KV shard to.
        from_rank = rank  # which SP worker to receive the sequence KV shard from.
        sid = rank  # start from the worker's own shard
        for _ in range(num_iters):
            to_rank = get_sequence_parallel_next_rank(to_rank)
            from_rank = get_sequence_parallel_prev_rank(from_rank)
            # Launch async communication operations
            if rank != from_rank:
                # reserve space for kv tensors received from other peers
                owned_shards[from_rank] = (
                    torch.empty(
                        get_k_shard_shape(batch_size, seq_len, from_rank, sp_size),
                        device=local_k.device,
                        dtype=local_k.dtype,
                    ),
                    torch.empty(
                        get_v_shard_shape(batch_size, seq_len, from_rank, sp_size),
                        device=local_v.device,
                        dtype=local_v.dtype,
                    ),
                )
            comm_reqs = self.launch_sp_comm_ops(
                owned_shards[from_rank], owned_shards[rank], from_rank, rank, to_rank
            )
            q_shard_stt, q_shard_end = get_sp_seq_range(seq_len, sid, sp_size)
            q_shard = q[:, q_shard_stt:q_shard_end]
            k_shard, v_shard = owned_shards[sid]
            # Ragged attention computation for self attention within the shard
            o, s = input_metadata.flashinfer_prefill_wrapper_ragged.forward_return_lse(
                q_shard.contiguous().view(-1, self.tp_q_head_num, self.head_dim),
                k_shard.contiguous().view(-1, self.tp_k_head_num, self.head_dim),
                v_shard.contiguous().view(-1, self.tp_v_head_num, self.head_dim),
                causal=True,
                sm_scale=self.scaling,
                logits_soft_cap=self.logit_cap,
            )
            append_merge_shard(output_shards[sid], o, s)
            # Paged attention computation for cross shard attention
            # NOTE: below schedule is for load balancing. Basically, at iteration i,
            # (i starting from 0), each SP worker will run i paged attentions.
            for existing_sid in owned_sids:
                if existing_sid == sid:
                    continue
                # Due to the causal nature of the attention, swap pids if necessary
                i, j = (
                    (existing_sid, sid) if existing_sid > sid else (sid, existing_sid)
                )
                q_shard_stt, q_shard_end = get_sp_seq_range(seq_len, i, sp_size)
                q_data = q[:, q_shard_stt:q_shard_end]
                # FIXME (yifan): should store them into kv cache and use kv cache here.
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

            # Wait for async communication to complete
            self.wait_sp_comm_ops(comm_reqs)
            if rank != from_rank:
                owned_sids.append(from_rank)
            sid = from_rank

        # Reshape all o tensors so that we can concatenate along the sequence dimension
        # we must have len(shard_list) == 1 here
        os = [
            o.view(batch_size, -1, self.tp_q_head_num, self.head_dim)
            for shard_list in output_shards
            for o, _ in shard_list
        ]
        o = torch.cat(os, dim=1)

        # FIXME (yifan): enable kv cache storage after we supoprt it.
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
