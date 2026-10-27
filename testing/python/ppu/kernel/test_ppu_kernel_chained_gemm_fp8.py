"""End-to-end FP8 chained GEMM using an in-register warp-shuffle relayout."""

import torch

import tilelang
import tilelang.ppu.language as T
import tilelang.testing


M = 16
K = 32
N = 16
FP8 = T.float8_e4m3fn


def _mma_c_layout():
    """PPU0015 FP32 accumulator layout for a 16x32 result tile."""
    return T.Fragment(
        (M, K),
        forward_thread_fn=lambda i, j: (i % 8) * 4 + (j % 8) // 2,
        forward_index_fn=lambda i, j: (j // 16) * 8 + ((j % 16) // 8) * 4 + (i // 8) * 2 + j % 2,
    )


def _mma_a_layout():
    """PPU0015 FP8 A-operand layout for a 16x32 tile."""
    return T.Fragment(
        (M, K),
        forward_thread_fn=lambda i, j: (i % 8) * 4 + (j % 16) // 4,
        forward_index_fn=lambda i, j: (j // 16) * 8 + (i // 8) * 4 + j % 4,
    )


@T.prim_func
def _chained_gemm(
    q: T.Tensor((M, K), FP8),
    k: T.Tensor((K, K), FP8),
    v: T.Tensor((N, K), FP8),
    output: T.Tensor((M, N), T.float32),
):
    # Q @ K.T produces (M, K); that FP8 fragment is then consumed by
    # probabilities @ V.T to produce (M, N).
    with T.Kernel(1, threads=32):
        q_shared = T.alloc_shared((M, K), FP8)
        k_shared = T.alloc_shared((K, K), FP8)
        v_shared = T.alloc_shared((N, K), FP8)
        scores = T.alloc_fragment((M, K), T.float32)
        probabilities = T.alloc_fragment((M, K), FP8)
        result = T.alloc_fragment((M, N), T.float32)
        score_registers = T.alloc_local((16,), T.float32)

        c_layout = _mma_c_layout()
        a_layout = _mma_a_layout()
        T.annotate_layout({scores: c_layout, probabilities: a_layout})

        T.copy(q, q_shared)
        T.copy(k, k_shared)
        T.copy(v, v_shared)
        T.clear(scores)
        T.gemm(q_shared, k_shared, scores, transpose_B=True)

        # Materialize the first GEMM's accumulator registers using their
        # physical per-lane indices, then express the C -> A ownership change
        # entirely with warp shuffles. T.bind marks the local index as a
        # physical per-thread slot rather than a distributed logical dimension.
        for i, j in T.Parallel(M, K, loop_layout=c_layout):
            score_slot = T.bind((j // 16) * 8 + ((j % 16) // 8) * 4 + (i // 8) * 2 + j % 2)
            score_registers[score_slot] = scores[i, j]

        lane = T.get_lane_idx()
        quad = lane % 4
        warp_row = lane // 4
        for i, j in T.Parallel(M, K, loop_layout=a_layout):
            row_half = i // 8
            atom = j // 16
            byte = j % 4
            source_lane = warp_row * 4 + (quad % 2) * 2 + byte // 2
            low_slot = T.bind(atom * 8 + row_half * 2 + byte % 2)
            high_slot = T.bind(atom * 8 + 4 + row_half * 2 + byte % 2)
            low = T.shfl_sync(score_registers[low_slot], source_lane)
            high = T.shfl_sync(score_registers[high_slot], source_lane)
            probabilities[i, j] = T.cast(T.if_then_else(quad < 2, low, high), FP8)

        T.clear(result)
        T.gemm(
            probabilities,
            v_shared,
            result,
            transpose_B=True,
            policy=T.GemmWarpPolicy.FullRow,
        )
        T.copy(result, output)


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_fp8_chained_gemm_with_packed_register_relayout():
    kernel = tilelang.compile(_chained_gemm, out_idx=[3])
    source = kernel.get_kernel_source()

    # Four logical FP8 values share each shuffled uint32 carrier, so the
    # relayout needs two shuffles and one byte permute per packed group.
    assert source.count("__shfl_sync") == 2
    assert source.count("__byte_perm") == 1
    assert source.count("tl::mma_sync") >= 2

    fp8_dtype = FP8.as_torch()
    for scale in (0.125, 0.25, 0.5):
        for seed in range(5):
            torch.manual_seed(seed)
            q = (torch.randn((M, K), device="cuda", dtype=torch.float16) * scale).to(fp8_dtype)
            k = (torch.randn((K, K), device="cuda", dtype=torch.float16) * scale).to(fp8_dtype)
            v = (torch.randn((N, K), device="cuda", dtype=torch.float16) * scale).to(fp8_dtype)

            actual = kernel(q, k, v)
            probabilities = (q.float() @ k.float().T).to(fp8_dtype)
            expected = probabilities.float() @ v.float().T
            # The reference explicitly applies the same intermediate FP8
            # quantization; the remaining tolerance covers MMA accumulation
            # order.  The maximum observed error for these cases is < 0.0021.
            torch.testing.assert_close(actual, expected, atol=5e-3, rtol=1e-3)


if __name__ == "__main__":
    tilelang.testing.main()
