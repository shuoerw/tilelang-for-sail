import torch

import tilelang
import tilelang.ppu.language as T
import tilelang.testing


@tilelang.testing.requires_ppu
def test_copy_and_async_copy_gemm_runtime_equivalent_ppu():
    """Check explicit async copy and planned T.copy agree on PPU."""

    M = 256
    N = 256
    K = 128
    block_M = 128
    block_N = 128
    block_K = 32

    @T.prim_func
    def matmul_relu_kernel(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M)) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.async_copy(A[by * block_M, ko * block_K], A_shared)
                T.ptx_wait_group(0)

                T.async_copy(B[ko * block_K, bx * block_N], B_shared)
                T.ptx_wait_group(0)

                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)

            T.copy(C_local, C[by * block_M, bx * block_N])

    async_matmul_relu = matmul_relu_kernel

    @T.prim_func
    def matmul_relu_kernel(  # noqa: F811
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M)) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), T.float16)
            B_shared = T.alloc_shared((block_K, block_N), T.float16)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)

            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, ko * block_K], A_shared)
                T.copy(B[ko * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)

            for i, j in T.Parallel(block_M, block_N):
                C_local[i, j] = T.max(C_local[i, j], 0)

            T.copy(C_local, C[by * block_M, bx * block_N])

    sync_matmul_relu = matmul_relu_kernel

    async_kernel = tilelang.compile(async_matmul_relu, target="ppu")
    sync_kernel = tilelang.compile(sync_matmul_relu, target="ppu")

    torch.manual_seed(0)
    a = torch.randn((M, K), device="cuda", dtype=torch.float16)
    b = torch.randn((K, N), device="cuda", dtype=torch.float16)
    async_result = torch.empty((M, N), device="cuda", dtype=torch.float16)
    sync_result = torch.empty_like(async_result)
    async_kernel(a, b, async_result)
    sync_kernel(a, b, sync_result)

    expected = torch.relu(a.float() @ b.float()).half()
    torch.testing.assert_close(async_result, sync_result, atol=1e-2, rtol=1e-2)
    torch.testing.assert_close(async_result, expected, atol=5e-2, rtol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
