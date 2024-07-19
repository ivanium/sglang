import pytest
import torch
from flashinfer import (
    BatchDecodeWithPagedKVCacheWrapper,
    BatchPrefillWithRaggedKVCacheWrapper,
    BatchPrefillWithPagedKVCacheWrapper,
)
from flashinfer.cascade import merge_state
from flashinfer.decode import _grouped_size_compiled_for_decode_kernels

from sglang.srt.layers.extend_attention import extend_attention_fwd, redundant_attention
from sglang.srt.layers.token_attention import token_attention_fwd

flashinfer_prefill_wrapper_ragged = None
flashinfer_prefill_wrapper_paged = None
flashinfer_decode_wrapper = None


@pytest.mark.parametrize("batch_size", [12, 37, 67])
@pytest.mark.parametrize("kv_len", [54, 97])
@pytest.mark.parametrize("qo_len", [37, 17])
@pytest.mark.parametrize("num_kv_heads", [4])
@pytest.mark.parametrize("num_qo_heads", [32, 4])
@pytest.mark.parametrize("head_dim", [128])
def test_seq_parallel_prefill(
    batch_size,
    kv_len,
    qo_len,
    num_kv_heads,
    num_qo_heads,
    head_dim,
):
    init_flashinfer(num_qo_heads, num_kv_heads)

    q = torch.randn(batch_size, qo_len, num_qo_heads, head_dim).to(0).half()
    k = torch.randn(batch_size, kv_len, num_kv_heads, head_dim).to(0).half()
    v = torch.randn(batch_size, kv_len, num_kv_heads, head_dim).to(0).half()

    def seq_parallel_worker_0_impl():
        num_partitions = 2
        num_iters = num_partitions
        qo_len_per_iter = qo_len // num_iters
        kv_len_per_partition = kv_len // num_partitions

        def iter0(i = 0): # SP worker 0 iter 0
            q0 = q[:, i * qo_len_per_iter : (i + 1) * qo_len_per_iter]
            k0 = k[:, i * kv_len_per_partition : (i + 1) * kv_len_per_partition]
            v0 = v[:, i * kv_len_per_partition : (i + 1) * kv_len_per_partition]

            qo_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len_per_iter
            kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * kv_len_per_partition
            flashinfer_prefill_wrapper_ragged.end_forward()
            flashinfer_prefill_wrapper_ragged.begin_forward(
                qo_indptr,
                kv_indptr,
                num_qo_heads,
                num_kv_heads,
                head_dim,
            )
            o00 = flashinfer_prefill_wrapper_ragged.forward(
                q0.contiguous().view(-1, num_qo_heads, head_dim),
                k0.contiguous().view(-1, num_kv_heads, head_dim),
                v0.contiguous().view(-1, num_kv_heads, head_dim),
            )
            flashinfer_prefill_wrapper_ragged.end_forward()
            return o00
        o00 = iter0()

        def iter1(i = 1): # SP worker 0 iter 1
            q1 = q[:, i * qo_len_per_iter : (i + 1) * qo_len_per_iter]
            k1 = k[:, i * kv_len_per_partition : (i + 1) * kv_len_per_partition]
            v1 = v[:, i * kv_len_per_partition : (i + 1) * kv_len_per_partition]

            qo_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len_per_iter
            kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * kv_len_per_partition

            # TODO: we don't really need to re-setup ragged attention
            flashinfer_prefill_wrapper_ragged.end_forward()
            flashinfer_prefill_wrapper_ragged.begin_forward(
                qo_indptr,
                kv_indptr,
                num_qo_heads,
                num_kv_heads,
                head_dim,
            )
            o11, s11 = flashinfer_prefill_wrapper_ragged.forward_return_lse(
                q1.contiguous().view(-1, num_qo_heads, head_dim),
                k1.contiguous().view(-1, num_kv_heads, head_dim),
                v1.contiguous().view(-1, num_kv_heads, head_dim),
            )
            flashinfer_prefill_wrapper_ragged.end_forward()

            k0 = k[:, (i - 1) * kv_len_per_partition : i * kv_len_per_partition]
            v0 = v[:, (i - 1) * kv_len_per_partition : i * kv_len_per_partition]
            kv_data0 = torch.zeros(batch_size * kv_len_per_partition, 2, num_kv_heads, head_dim).to(0).half()
            kv_data0[:, 0] = k0.contiguous().view(-1, num_kv_heads, head_dim)
            kv_data0[:, 1] = v0.contiguous().view(-1, num_kv_heads, head_dim)

            kv_indices = torch.arange(0, batch_size * kv_len_per_partition).to(0).int()
            kv_last_page_len = torch.full((batch_size,), 1, dtype=torch.int32).to(0)

            flashinfer_prefill_wrapper_paged.end_forward()
            flashinfer_prefill_wrapper_paged.begin_forward(
                qo_indptr,
                kv_indptr,
                kv_indices,
                kv_last_page_len,
                num_qo_heads,
                num_kv_heads,
                head_dim,
                1,
            )
            o10, s10 = flashinfer_prefill_wrapper_paged.forward_return_lse(
                q1.contiguous().view(-1, num_qo_heads, head_dim), kv_data0,
                causal=False,
            )
            flashinfer_prefill_wrapper_paged.end_forward()

            o1, _ = merge_state(o10, s10, o11, s11)
            return o1

        o1 = iter1()
        o = torch.cat([
                o00.view(batch_size, qo_len_per_iter, num_qo_heads, head_dim),
                o1.view(batch_size, qo_len_per_iter, num_qo_heads, head_dim),
            ], dim=1)
        return o.view(-1, num_qo_heads, head_dim)

    def reference_impl_ragged():
        qo_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len
        kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * kv_len

        flashinfer_prefill_wrapper_ragged.end_forward()
        flashinfer_prefill_wrapper_ragged.begin_forward(
            qo_indptr,
            kv_indptr,
            num_qo_heads,
            num_kv_heads,
            head_dim,
        )
        o = flashinfer_prefill_wrapper_ragged.forward(
            q.contiguous().view(-1, num_qo_heads, head_dim),
            k.contiguous().view(-1, num_kv_heads, head_dim),
            v.contiguous().view(-1, num_kv_heads, head_dim),
        )
        flashinfer_prefill_wrapper_ragged.end_forward()
        return o

    def reference_impl_paged():
        qo_indptr = torch.arange(0, batch_size + 1).to(0).int() * qo_len
        total_tokens = kv_len * batch_size

        kv_data = torch.zeros(total_tokens, 2, num_kv_heads, head_dim).to(0).half()
        kv_data[:, 0] = k.contiguous().view(-1, num_kv_heads, head_dim)
        kv_data[:, 1] = v.contiguous().view(-1, num_kv_heads, head_dim)
        kv_indptr = torch.arange(0, batch_size + 1).to(0).int() * kv_len
        kv_indices = torch.arange(0, total_tokens).to(0).int()
        kv_last_page_len = torch.full((batch_size,), 1, dtype=torch.int32).to(0)

        flashinfer_prefill_wrapper_paged.end_forward()
        flashinfer_prefill_wrapper_paged.begin_forward(
            qo_indptr,
            kv_indptr,
            kv_indices,
            kv_last_page_len,
            num_qo_heads,
            num_kv_heads,
            head_dim,
            1,
        )
        o = flashinfer_prefill_wrapper_paged.forward(
            q.contiguous().view(-1, num_qo_heads, head_dim), kv_data
        )
        flashinfer_prefill_wrapper_paged.end_forward()
        return o

    o_sp = seq_parallel_worker_0_impl()
    o_truth = reference_impl_paged()

    print("Mean: ", torch.mean(torch.abs(o_sp - o_truth)))
    print("Max: ", torch.max(torch.abs(o_sp - o_truth)))
    assert torch.allclose(o_sp, o_truth, rtol=1e-2, atol=1e-3)


def init_flashinfer(num_attention_heads, num_kv_heads):
    if not _grouped_size_compiled_for_decode_kernels(num_attention_heads, num_kv_heads):
        use_tensor_cores = True
    else:
        use_tensor_cores = False

    workspace_buffer = torch.empty(128 * 1024 * 1024, dtype=torch.int8, device="cuda")

    global flashinfer_prefill_wrapper_ragged, flashinfer_prefill_wrapper_paged, flashinfer_decode_wrapper

    flashinfer_prefill_wrapper_ragged = BatchPrefillWithRaggedKVCacheWrapper(
        workspace_buffer, "NHD"
    )
    flashinfer_prefill_wrapper_paged = BatchPrefillWithPagedKVCacheWrapper(
        workspace_buffer, "NHD"
    )
    flashinfer_decode_wrapper = BatchDecodeWithPagedKVCacheWrapper(
        workspace_buffer, "NHD", use_tensor_cores=use_tensor_cores
    )


if __name__ == "__main__":
    test_seq_parallel_prefill(12, 128, 128, 8, 8, 128)
    test_seq_parallel_prefill(12, 4096, 4096, 8, 8, 128)
    test_seq_parallel_prefill(12, 1024, 1024, 32, 32, 128)