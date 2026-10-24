"""Verify that PPU15 GEMM uses swizzled ldmatrix by default."""

import torch

import tilelang
import tilelang.language as T
import tilelang.testing


M = N = K = 256


def _matmul_bf16(block_M, block_N, block_K, transpose_b):
    dtype = T.bfloat16
    accum_dtype = T.float32
    b_shape = (N, K) if transpose_b else (K, N)

    @T.prim_func
    def main(A: T.Tensor((M, K), dtype), B: T.Tensor(b_shape, dtype), C: T.Tensor((M, N), accum_dtype)):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K) if transpose_b else (block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                if transpose_b:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=transpose_b)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_gemm_bf16_nt_ldmat_swzl_default():
    kernel = tilelang.compile(_matmul_bf16(128, 128, 64, True), out_idx=[-1])
    source = kernel.get_kernel_source()

    assert "tl::tix_ldmatrix_swzl_x4(" in source
    assert "__builtin_ppu_to_uniform_b32(" in source

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(N, K, device="cuda", dtype=torch.bfloat16)
    c = kernel(a, b)
    ref = a.float() @ b.float().T
    torch.testing.assert_close(c, ref, rtol=2e-2, atol=2e-2)


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_gemm_bf16_nn_ldmat_swzl_default():
    kernel = tilelang.compile(_matmul_bf16(128, 128, 64, False), out_idx=[-1])
    source = kernel.get_kernel_source()

    assert "tl::tix_ldmatrix_swzl_x4(" in source
    assert "tl::tix_ldmatrix_swzl_x4_trans(" in source
    assert "__builtin_ppu_to_uniform_b32(" in source

    a = torch.randn(M, K, device="cuda", dtype=torch.bfloat16)
    b = torch.randn(K, N, device="cuda", dtype=torch.bfloat16)
    c = kernel(a, b)
    ref = a.float() @ b.float()
    torch.testing.assert_close(c, ref, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    tilelang.testing.main()
