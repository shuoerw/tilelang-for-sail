"""GEMM register pipelining is automatic when the inner loop has multiple tiles."""

import pytest
import torch

import tilelang
import tilelang.ppu.language as T
import tilelang.testing


M = N = 128
K = 256


def _matmul_bf16(block_k):
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, block_k), dtype)
            B_shared = T.alloc_shared((block_k, 128), dtype)
            C_local = T.alloc_fragment((128, 128), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_k), num_stages=2):
                T.copy(A[by * 128, k * block_k], A_shared)
                T.copy(B[k * block_k, bx * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * 128, bx * 128])

    return main


@tilelang.testing.requires_ppu_compute_version(1, 5)
@pytest.mark.parametrize("block_k", [16, 64])
def test_gemm_pipeline_default_and_single_tile_fallback(block_k):
    kernel = tilelang.compile(_matmul_bf16(block_k), out_idx=[-1])
    source = kernel.get_kernel_source()

    for operand in ("A", "B"):
        # Outer-pipeline epilogues can rename ordinary buffers to *_local_1.
        assert (f"{operand}_local_0[" in source) == (block_k > 16)
        if block_k > 16:
            assert f"{operand}_local_1[" in source

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    c = kernel(a, b)
    ref = a.float() @ b.float()
    torch.testing.assert_close(c, ref, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    tilelang.testing.main()
