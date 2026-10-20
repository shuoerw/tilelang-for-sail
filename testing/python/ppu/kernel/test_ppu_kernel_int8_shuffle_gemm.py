"""End-to-end PPU0015 INT8 GEMM fed by a packed register relayout."""

import re

import torch

import tilelang
import tilelang.ppu.language as T
import tilelang.testing


M = 16
N = 16
K = 32
DP4A_M = 4
DP4A_N = 8


def _mma_a_layout():
    """PPU0015 INT8 A-operand layout for one m16n16k32 atom."""
    return T.Fragment(
        (M, K),
        forward_thread_fn=lambda i, j: 4 * (i % 8) + (j % 16) // 4,
        forward_index_fn=lambda i, j: 8 * (j // 16) + (i // 8) * 4 + j % 4,
    )


@T.prim_func
def _int8_shuffle_gemm(
    a: T.Tensor((M, K), T.int8),
    b: T.Tensor((N, K), T.int8),
    output: T.Tensor((M, N), T.int32),
):
    with T.Kernel(1, threads=32):
        b_shared = T.alloc_shared((N, K), T.int8)
        a_fragment = T.alloc_fragment((M, K), T.int8)
        result = T.alloc_fragment((M, N), T.int32)
        a_layout = _mma_a_layout()
        T.annotate_layout({a_fragment: a_layout})

        T.copy(b, b_shared)
        lane = T.get_lane_idx()
        for i, j in T.Parallel(M, K, loop_layout=a_layout):
            a_fragment[i, j] = T.cast(T.shfl_sync(T.cast(a[i, j], T.int32), lane), T.int8)

        T.clear(result)
        T.gemm(a_fragment, b_shared, result, transpose_B=True)
        T.copy(result, output)


@T.prim_func
def _int8_shuffle_dp4a(
    a: T.Tensor((DP4A_M, K), T.int8),
    b: T.Tensor((DP4A_N, K), T.int8),
    output: T.Tensor((DP4A_M, DP4A_N), T.int32),
):
    with T.Kernel(1, threads=32):
        a_local = T.alloc_local((K,), T.int8)
        b_local = T.alloc_local((K,), T.int8)
        accum = T.alloc_local((1,), T.int32)
        lane = T.get_lane_idx()
        row = lane // DP4A_N
        col = lane % DP4A_N

        for k in T.serial(K):
            a_local[k] = T.cast(T.shfl_sync(T.cast(a[row, k], T.int32), lane), T.int8)
            b_local[k] = b[col, k]

        accum[0] = 0
        for k4 in T.serial(K // 4):
            T.dp4a(a_local[k4 * 4], b_local[k4 * 4], accum[0])
        output[row, col] = accum[0]


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu15_int8_gemm_with_packed_single_source_lane_shuffle():
    kernel = tilelang.compile(_int8_shuffle_gemm, out_idx=[2])
    source = kernel.get_kernel_source()
    loop_extents = [int(extent) for extent in re.findall(r"for \(int [^;]+; [^<]+< (\d+);", source)]

    # Four logical INT8 values share each shuffled uint32 carrier.  A scalar
    # fallback would retain a 16-iteration relayout loop instead.
    assert source.count("__shfl_sync") == 1
    assert 4 in loop_extents
    assert 16 not in loop_extents
    assert source.count("tl::mma_sync") == 1

    for seed in range(5):
        torch.manual_seed(seed)
        a = torch.randint(-4, 5, (M, K), device="cuda", dtype=torch.int8)
        b = torch.randint(-4, 5, (N, K), device="cuda", dtype=torch.int8)
        actual = kernel(a, b)
        expected = (a.cpu().to(torch.int32) @ b.cpu().to(torch.int32).T).to(device="cuda", dtype=torch.int32)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_eq(1, 0)
def test_ppu10_int8_dp4a_with_packed_single_source_lane_shuffle():
    kernel = tilelang.compile(_int8_shuffle_dp4a, out_idx=[2])
    source = kernel.get_kernel_source()

    assert source.count("__shfl_sync") == 1
    assert len(re.findall(r"DP4A|dp4a", source)) >= 1

    for seed in range(5):
        torch.manual_seed(seed)
        a = torch.randint(-4, 5, (DP4A_M, K), device="cuda", dtype=torch.int8)
        b = torch.randint(-4, 5, (DP4A_N, K), device="cuda", dtype=torch.int8)
        actual = kernel(a, b)
        expected = (a.cpu().to(torch.int32) @ b.cpu().to(torch.int32).T).to(device="cuda", dtype=torch.int32)
        torch.testing.assert_close(actual, expected, atol=0, rtol=0)


if __name__ == "__main__":
    tilelang.testing.main()
