"""Verify that the GEMM register ping-pong pipeline is opt-in."""

import torch

import tilelang
import tilelang.language as T
import tilelang.testing


M = N = 128
K = 256


def _matmul_bf16(pipeline):
    dtype = T.bfloat16
    accum_dtype = T.float32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 64), dtype)
            B_shared = T.alloc_shared((64, 128), dtype)
            C_local = T.alloc_fragment((128, 128), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, 64), num_stages=2):
                T.copy(A[by * 128, k * 64], A_shared)
                T.copy(B[k * 64, bx * 128], B_shared)
                T.gemm(A_shared, B_shared, C_local, pipeline=pipeline)
            T.copy(C_local, C[by * 128, bx * 128])

    return main


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_gemm_pipeline_default_off():
    kernel = tilelang.compile(_matmul_bf16(False), out_idx=[-1])
    source = kernel.get_kernel_source()

    assert "A_local_0[" not in source
    assert "B_local_0[" not in source

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    c = kernel(a, b)
    ref = a.float() @ b.float()
    torch.testing.assert_close(c, ref, rtol=2e-2, atol=2e-2)


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_gemm_pipeline_explicit_on():
    kernel = tilelang.compile(_matmul_bf16(True), out_idx=[-1])
    source = kernel.get_kernel_source()

    assert "A_local_0[" in source
    assert "A_local_1[" in source
    assert "B_local_0[" in source
    assert "B_local_1[" in source

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    c = kernel(a, b)
    ref = a.float() @ b.float()
    torch.testing.assert_close(c, ref, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    tilelang.testing.main()
