import torch
import tilelang
import tilelang.ppu.language as T
from tilelang.language.fp8 import determine_fp8_type


def calc_diff(x, y):
    x, y = x.double(), y.double()
    denominator = (x * x + y * y).sum()
    sim = 2 * (x * y).sum() / denominator
    return 1 - sim


def _make_matmul(pass_configs=None):
    @tilelang.jit(pass_configs=pass_configs)
    def matmul(A, B, block_M, block_N, block_K, dtype, trans_a=False, trans_b=True, accum_dtype=T.float32):
        M, N, K = T.const("M, N, K")

        # Logical GEMM: C[M, N] = A_logical[M, K] @ B_logical[K, N]
        # trans_a=False: A is stored as (M, K); trans_a=True: A is stored as (K, M)
        A: T.Tensor((K, M) if trans_a else (M, K), dtype)
        # trans_b=False: B is stored as (K, N); trans_b=True: B is stored as (N, K)
        B: T.Tensor((N, K) if trans_b else (K, N), dtype)
        C = T.empty((M, N), dtype)

        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_K, block_M) if trans_a else (block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K) if trans_b else (block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                if trans_a:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_b:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_A=trans_a, transpose_B=trans_b)

            T.copy(C_local, C[by * block_M, bx * block_N])

        return C

    return matmul


# AIU global->shared bulk copy and swizzled ldmatrix (ldmat.swzl) both enabled
matmul = _make_matmul(
    {
        tilelang.PassConfigKey.TL_DISABLE_AIU_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_LDMAT_SWZL: False,
    }
)
# Default pass configs: AIU lowering and ldmat.swzl both disabled
matmul_default = _make_matmul()


def test_gemm_fp8(M, N, K, dtype, trans_a=False, trans_b=True, use_default_configs=False):
    tag = ("T" if trans_a else "N") + ("T" if trans_b else "N")
    if use_default_configs:
        tag += "-default"
    torch_dtype = T.dtype(dtype).as_torch()

    a = torch.randn(*((K, M) if trans_a else (M, K)), dtype=torch.float16, device="cuda").to(dtype=torch_dtype)
    b = torch.randn(*((N, K) if trans_b else (K, N)), dtype=torch.float16, device="cuda").to(dtype=torch_dtype)

    kernel = matmul_default if use_default_configs else matmul
    c = kernel(a, b, 128, 128, 64, dtype, trans_a, trans_b)

    a_logical = a.half().t() if trans_a else a.half()
    b_logical = b.half().t() if trans_b else b.half()
    ref_c = (a_logical @ b_logical).to(dtype=torch_dtype)

    print(f"[{tag}]")
    print(c)
    print(ref_c)

    diff = calc_diff(c, ref_c)
    print(f"[{tag}] diff: {diff}")
    assert diff < 1e-3


def main():
    # Cover all transpose combinations: NT / TT / TN / NN
    for trans_a, trans_b in [(False, True), (True, True), (True, False), (False, False)]:
        for use_default_configs in (False, True):
            test_gemm_fp8(1024, 1024, 1024, determine_fp8_type(), trans_a, trans_b, use_default_configs)
            test_gemm_fp8(1024, 1024, 1024, determine_fp8_type("e5m2"), trans_a, trans_b, use_default_configs)


def run_regression_perf():
    M, N, K = 4096, 4096, 4096
    dtype = determine_fp8_type()
    kernel_e4m3 = matmul.compile(M=M, N=N, K=K, block_M=128, block_N=128, block_K=64, dtype=dtype)
    profiler_e4m3 = kernel_e4m3.get_profiler(tilelang.TensorSupplyType.Integer)
    latency_e4m3 = profiler_e4m3.do_bench(backend="cupti")

    dtype = determine_fp8_type("e5m2")
    kernel_e5m2 = matmul.compile(M=M, N=N, K=K, block_M=128, block_N=128, block_K=64, dtype=dtype)
    profiler_e5m2 = kernel_e5m2.get_profiler(tilelang.TensorSupplyType.Integer)
    latency_e5m2 = profiler_e5m2.do_bench(backend="cupti")
    print(f"e4m3: {latency_e4m3}, e5m2: {latency_e5m2}")
    return (latency_e4m3 + latency_e5m2) / 2


if __name__ == "__main__":
    main()
