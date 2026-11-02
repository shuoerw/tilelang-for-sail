import tilelang
import tilelang.ppu.language as T
from tilelang.transform import PassConfigKey


@tilelang.jit(pass_configs={PassConfigKey.TL_DISABLE_AIU_LOWER: False, PassConfigKey.TL_DISABLE_LDMAT_SWZL: False})
def matmul(A, B, block_M, block_N, block_K, num_stages, dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


def main():
    import torch

    test_configs = [
        {"num_stages": 3},  # with software pipeline
        {"num_stages": 0},  # without software pipeline
    ]

    for config in test_configs:
        num_stages = config["num_stages"]
        print(f"\n{'=' * 60}")
        print(f"Testing num_stages={num_stages}")
        print(f"{'=' * 60}")

        kernel = matmul.compile(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32, num_stages=num_stages)

        a = torch.randn(1024, 1024).cuda().half()
        b = torch.randn(1024, 1024).cuda().half()

        c = kernel(a, b)

        ref_c = a @ b

        print("c:")
        print(c)
        print("ref_c:")
        print(ref_c)

        torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
        print(f"All check passed. (num_stages={num_stages})")

        # Get CUDA Source
        print("CUDA Source:")
        print(kernel.get_kernel_source())

        # benchmark
        profiler = kernel.get_profiler()
        latency = profiler.do_bench(backend="cupti")
        # latency = profiler.do_bench()
        print(f"tilelang Latency: {latency}ms")


def run_regression_perf():
    kernel = matmul.compile(M=1024, N=1024, K=1024, block_M=128, block_N=128, block_K=32, num_stages=3)
    profiler = kernel.get_profiler()
    return profiler.do_bench(backend="cupti")


if __name__ == "__main__":
    main()
