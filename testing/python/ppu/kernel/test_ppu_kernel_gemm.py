"""PPU GEMM kernel tests.

Covers SS (shared-shared), SR (shared-register), and RS (register-shared)
layout variants with all PPU-supported dtypes (float16, bfloat16, int8,
tfloat32, float32, float8 e4m3/e5m2) and NN/TN/NT transpose combinations.
"""

import pytest

from tilelang import tvm as tvm
import tilelang
import tilelang.testing
import tilelang.language as T


def _is_fp8_dtype(dtype) -> bool:
    return "float8" in str(dtype)


def _gen_fp8_inputs(A_shape, B_shape, in_dtype, b_in_dtype):
    """Generate finite fp8 inputs: sample in fp16 first, then cast, so that
    the default random-bit supply does not produce NaN/Inf bit patterns."""
    import torch

    from tilelang.utils.device import get_current_device

    device = get_current_device()
    A = torch.randn(*A_shape, dtype=torch.float16, device=device).to(in_dtype.as_torch())
    B = torch.randn(*B_shape, dtype=torch.float16, device=device).to(b_in_dtype.as_torch())
    return [A, B]


# ---------------------------------------------------------------------------
# SS variant: A and B both in shared memory
# ---------------------------------------------------------------------------


def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
    b_in_dtype=None,
    pipeline=False,
):
    if b_in_dtype is None:
        b_in_dtype = in_dtype
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, b_in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, b_in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B, pipeline=pipeline)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=0,
    num_threads=128,
    b_in_dtype=None,
    expected_source_counts=None,
):
    program = matmul(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
        b_in_dtype,
    )

    kernel = tilelang.compile(program, out_idx=[2])
    if expected_source_counts is not None:
        source = kernel.get_kernel_source()
        for snippet, expected_count in expected_source_counts.items():
            assert source.count(snippet) == expected_count
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        if in_dtype in (T.tfloat32, T.float32):
            # Convert float32 to tfloat32 because tfloat32 mma cannot truncate
            # float32 automatically, -0x1000 means
            A = (A.view(torch.int32) - 0x1000).view(torch.float32)
            B = (B.view(torch.int32) - 0x1000).view(torch.float32)
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    _b = b_in_dtype if b_in_dtype is not None else in_dtype
    if _is_fp8_dtype(in_dtype) or _is_fp8_dtype(_b):
        ins = _gen_fp8_inputs(A_shape, B_shape, in_dtype, _b)
        profiler.assert_allclose(ref_program, input_tensors=ins, atol=1e-2, rtol=1e-2)
    else:
        profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_f16f16f16_nn():
    run_gemm(
        512,
        1024,
        768,
        False,
        False,
        T.float16,
        T.float16,
        T.float16,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f16f16f32_nn():
    run_gemm(
        512,
        1024,
        768,
        False,
        False,
        T.float16,
        T.float16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f16f16f32_tn():
    run_gemm(
        512,
        1024,
        768,
        True,
        False,
        T.float16,
        T.float16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f16f16f32_nt():
    run_gemm(
        512,
        1024,
        768,
        False,
        True,
        T.float16,
        T.float16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_bf16bf16f32_nn():
    run_gemm(
        512,
        1024,
        768,
        False,
        False,
        T.bfloat16,
        T.bfloat16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_bf16bf16f32_tn():
    run_gemm(
        512,
        1024,
        768,
        True,
        False,
        T.bfloat16,
        T.bfloat16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_bf16bf16f32_nt():
    run_gemm(
        512,
        1024,
        768,
        False,
        True,
        T.bfloat16,
        T.bfloat16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_i8i8i32_nn():
    run_gemm(
        512,
        1024,
        768,
        False,
        False,
        T.int8,
        T.int8,
        T.int32,
        128,
        128,
        64,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_eq(1, 0)
def test_ppu10_gemm_i8i8i32_m16n16k32_nt():
    run_gemm(
        16,
        16,
        32,
        False,
        True,
        T.int8,
        T.int32,
        T.int32,
        16,
        16,
        32,
        num_stages=0,
        num_threads=32,
        expected_source_counts={
            "tl::tix_ldmatrix_x4(": 2,
            "tl::tix_ldmatrix_x4_native_ppu10(": 0,
        },
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_eq(1, 0)
@pytest.mark.parametrize("trans_B", [False, True])
def test_ppu10_gemm_i8i8i32_m16n16k32_transposed_a_rejected(trans_B):
    program = matmul(
        16,
        16,
        32,
        16,
        16,
        32,
        True,
        trans_B,
        T.int8,
        T.int32,
        T.int32,
        num_stages=0,
        threads=32,
    )
    with pytest.raises(ValueError, match="Unsupported k_dim 32"):
        tilelang.compile(program, out_idx=[2])


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_f16f16f16_tn():
    run_gemm(
        512,
        1024,
        768,
        True,
        False,
        T.float16,
        T.float16,
        T.float16,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_f16f16f16_nt():
    run_gemm(
        512,
        1024,
        768,
        False,
        True,
        T.float16,
        T.float16,
        T.float16,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_i8i8i32_tn():
    run_gemm(
        512,
        1024,
        768,
        True,
        False,
        T.int8,
        T.int8,
        T.int32,
        128,
        128,
        64,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_i8i8i32_nt():
    run_gemm(
        512,
        1024,
        768,
        False,
        True,
        T.int8,
        T.int8,
        T.int32,
        128,
        128,
        64,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_tf32f32f32_nn():
    run_gemm(
        512,
        1024,
        768,
        False,
        False,
        T.tfloat32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_tf32f32f32_tn():
    run_gemm(
        512,
        1024,
        768,
        True,
        False,
        T.tfloat32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_tf32f32f32_nt():
    run_gemm(
        512,
        1024,
        768,
        False,
        True,
        T.tfloat32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f32f32f32_nn():
    run_gemm(
        512,
        1024,
        768,
        False,
        False,
        T.float32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f32f32f32_tn():
    run_gemm(
        512,
        1024,
        768,
        True,
        False,
        T.float32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f32f32f32_nt():
    run_gemm(
        512,
        1024,
        768,
        False,
        True,
        T.float32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e4m3f32_nn():
    run_gemm(
        512,
        1024,
        768,
        False,
        False,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e4m3f32_tn():
    run_gemm(
        512,
        1024,
        768,
        True,
        False,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e4m3f32_nt():
    run_gemm(
        512,
        1024,
        768,
        False,
        True,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e5m2f32_nn():
    run_gemm(
        512,
        1024,
        768,
        False,
        False,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e5m2f32_tn():
    run_gemm(
        512,
        1024,
        768,
        True,
        False,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e5m2f32_nt():
    run_gemm(
        512,
        1024,
        768,
        False,
        True,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e5m2f32_nn():
    run_gemm(
        512,
        1024,
        768,
        False,
        False,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e5m2,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e4m3f32_nn():
    run_gemm(
        512,
        1024,
        768,
        False,
        False,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e4m3fn,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e5m2f32_tn():
    run_gemm(
        512,
        1024,
        768,
        True,
        False,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e5m2,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e5m2f32_nt():
    run_gemm(
        512,
        1024,
        768,
        False,
        True,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e5m2,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e4m3f32_tn():
    run_gemm(
        512,
        1024,
        768,
        True,
        False,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e4m3fn,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e4m3f32_nt():
    run_gemm(
        512,
        1024,
        768,
        False,
        True,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e4m3fn,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_pad_f16f16f16_nn():
    run_gemm(
        512 - 9,
        1024 - 7,
        768 - 5,
        False,
        False,
        T.float16,
        T.float16,
        T.float16,
        128,
        256,
        32,
        2,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_pad_f16f16f32_nn():
    run_gemm(
        512 - 9,
        1024 - 7,
        768 - 5,
        False,
        False,
        T.float16,
        T.float16,
        T.float32,
        128,
        256,
        32,
        2,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_f16f16f16_tt():
    run_gemm(
        512,
        1024,
        768,
        True,
        True,
        T.float16,
        T.float16,
        T.float16,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f16f16f32_tt():
    run_gemm(
        512,
        1024,
        768,
        True,
        True,
        T.float16,
        T.float16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_bf16bf16f32_tt():
    run_gemm(
        512,
        1024,
        768,
        True,
        True,
        T.bfloat16,
        T.bfloat16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_i8i8i32_tt():
    run_gemm(
        512,
        1024,
        768,
        True,
        True,
        T.int8,
        T.int8,
        T.int32,
        128,
        128,
        64,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_tf32f32f32_tt():
    run_gemm(
        512,
        1024,
        768,
        True,
        True,
        T.tfloat32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f32f32f32_tt():
    run_gemm(
        512,
        1024,
        768,
        True,
        True,
        T.float32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e4m3f32_tt():
    run_gemm(
        512,
        1024,
        768,
        True,
        True,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e5m2f32_tt():
    run_gemm(
        512,
        1024,
        768,
        True,
        True,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e5m2f32_tt():
    run_gemm(
        512,
        1024,
        768,
        True,
        True,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e5m2,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e4m3f32_tt():
    run_gemm(
        512,
        1024,
        768,
        True,
        True,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e4m3fn,
    )


# ---------------------------------------------------------------------------
# SR variant: A in shared, B in registers (fragment)
# ---------------------------------------------------------------------------


def matmul_sr(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
    b_in_dtype=None,
):
    if b_in_dtype is None:
        b_in_dtype = in_dtype
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, b_in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, b_in_dtype)
            B_local = T.alloc_fragment(B_shared_shape, b_in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                    T.copy(B_shared, B_local)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                    T.copy(B_shared, B_local)
                T.gemm(A_shared, B_local, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_sr(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=1,
    num_threads=128,
    b_in_dtype=None,
):
    program = matmul_sr(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
        b_in_dtype,
    )

    kernel = tilelang.compile(program, out_idx=[2])
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        if in_dtype in (T.tfloat32, T.float32):
            # Convert float32 to tfloat32 because tfloat32 mma cannot truncate
            # float32 automatically, -0x1000 means
            A = (A.view(torch.int32) - 0x1000).view(torch.float32)
            B = (B.view(torch.int32) - 0x1000).view(torch.float32)
        A = A.to(torch.float)
        B = B.to(torch.float)
        C = torch.matmul(A, B)
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    _b = b_in_dtype if b_in_dtype is not None else in_dtype
    if _is_fp8_dtype(in_dtype) or _is_fp8_dtype(_b):
        ins = _gen_fp8_inputs(A_shape, B_shape, in_dtype, _b)
        profiler.assert_allclose(ref_program, input_tensors=ins, atol=1e-2, rtol=1e-2)
    else:
        profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_f16f16f16_sr_nt():
    run_gemm_sr(
        512,
        1024,
        768,
        False,
        True,
        T.float16,
        T.float16,
        T.float16,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f16f16f32_sr_nn():
    run_gemm_sr(
        512,
        1024,
        768,
        False,
        False,
        T.float16,
        T.float16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f16f16f32_sr_tn():
    run_gemm_sr(
        512,
        1024,
        768,
        True,
        False,
        T.float16,
        T.float16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f16f16f32_sr_nt():
    run_gemm_sr(
        512,
        1024,
        768,
        False,
        True,
        T.float16,
        T.float16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_bf16bf16f32_sr_nt():
    run_gemm_sr(
        512,
        1024,
        768,
        False,
        True,
        T.bfloat16,
        T.bfloat16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_bf16bf16f32_sr_nn():
    run_gemm_sr(
        512,
        1024,
        768,
        False,
        False,
        T.bfloat16,
        T.bfloat16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_bf16bf16f32_sr_tn():
    run_gemm_sr(
        512,
        1024,
        768,
        True,
        False,
        T.bfloat16,
        T.bfloat16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_f16f16f16_sr_nn():
    run_gemm_sr(
        512,
        1024,
        768,
        False,
        False,
        T.float16,
        T.float16,
        T.float16,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_f16f16f16_sr_tn():
    run_gemm_sr(
        512,
        1024,
        768,
        True,
        False,
        T.float16,
        T.float16,
        T.float16,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_i8i8i32_sr_nt():
    run_gemm_sr(
        512,
        1024,
        768,
        False,
        True,
        T.int8,
        T.int8,
        T.int32,
        128,
        128,
        64,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_i8i8i32_sr_nn():
    run_gemm_sr(
        512,
        1024,
        768,
        False,
        False,
        T.int8,
        T.int8,
        T.int32,
        128,
        128,
        64,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_i8i8i32_sr_tn():
    run_gemm_sr(
        512,
        1024,
        768,
        True,
        False,
        T.int8,
        T.int8,
        T.int32,
        128,
        128,
        64,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_tf32f32f32_sr_nn():
    run_gemm_sr(
        512,
        1024,
        768,
        False,
        False,
        T.tfloat32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_tf32f32f32_sr_tn():
    run_gemm_sr(
        512,
        1024,
        768,
        True,
        False,
        T.tfloat32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_tf32f32f32_sr_nt():
    run_gemm_sr(
        512,
        1024,
        768,
        False,
        True,
        T.tfloat32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f32f32f32_sr_nn():
    run_gemm_sr(
        512,
        1024,
        768,
        False,
        False,
        T.float32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f32f32f32_sr_tn():
    run_gemm_sr(
        512,
        1024,
        768,
        True,
        False,
        T.float32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f32f32f32_sr_nt():
    run_gemm_sr(
        512,
        1024,
        768,
        False,
        True,
        T.float32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e4m3f32_sr_nn():
    run_gemm_sr(
        512,
        1024,
        768,
        False,
        False,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e4m3f32_sr_tn():
    run_gemm_sr(
        512,
        1024,
        768,
        True,
        False,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e4m3f32_sr_nt():
    run_gemm_sr(
        512,
        1024,
        768,
        False,
        True,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e5m2f32_sr_nn():
    run_gemm_sr(
        512,
        1024,
        768,
        False,
        False,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e5m2f32_sr_tn():
    run_gemm_sr(
        512,
        1024,
        768,
        True,
        False,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e5m2f32_sr_nt():
    run_gemm_sr(
        512,
        1024,
        768,
        False,
        True,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e5m2f32_sr_nn():
    run_gemm_sr(
        512,
        1024,
        768,
        False,
        False,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e5m2,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e4m3f32_sr_nn():
    run_gemm_sr(
        512,
        1024,
        768,
        False,
        False,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e4m3fn,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e5m2f32_sr_tn():
    run_gemm_sr(
        512,
        1024,
        768,
        True,
        False,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e5m2,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e5m2f32_sr_nt():
    run_gemm_sr(
        512,
        1024,
        768,
        False,
        True,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e5m2,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e4m3f32_sr_tn():
    run_gemm_sr(
        512,
        1024,
        768,
        True,
        False,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e4m3fn,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e4m3f32_sr_nt():
    run_gemm_sr(
        512,
        1024,
        768,
        False,
        True,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e4m3fn,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_f16f16f16_sr_tt():
    run_gemm_sr(
        512,
        1024,
        768,
        True,
        True,
        T.float16,
        T.float16,
        T.float16,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f16f16f32_sr_tt():
    run_gemm_sr(
        512,
        1024,
        768,
        True,
        True,
        T.float16,
        T.float16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_bf16bf16f32_sr_tt():
    run_gemm_sr(
        512,
        1024,
        768,
        True,
        True,
        T.bfloat16,
        T.bfloat16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_i8i8i32_sr_tt():
    run_gemm_sr(
        512,
        1024,
        768,
        True,
        True,
        T.int8,
        T.int8,
        T.int32,
        128,
        128,
        64,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_tf32f32f32_sr_tt():
    run_gemm_sr(
        512,
        1024,
        768,
        True,
        True,
        T.tfloat32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f32f32f32_sr_tt():
    run_gemm_sr(
        512,
        1024,
        768,
        True,
        True,
        T.float32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e4m3f32_sr_tt():
    run_gemm_sr(
        512,
        1024,
        768,
        True,
        True,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e5m2f32_sr_tt():
    run_gemm_sr(
        512,
        1024,
        768,
        True,
        True,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e5m2f32_sr_tt():
    run_gemm_sr(
        512,
        1024,
        768,
        True,
        True,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e5m2,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e4m3f32_sr_tt():
    run_gemm_sr(
        512,
        1024,
        768,
        True,
        True,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e4m3fn,
    )


# ---------------------------------------------------------------------------
# RS variant: A in registers (fragment), B in shared
# ---------------------------------------------------------------------------


def matmul_rs(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
    b_in_dtype=None,
):
    if b_in_dtype is None:
        b_in_dtype = in_dtype
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, b_in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype, scope="shared")
            A_local = T.alloc_fragment(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, b_in_dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                    T.copy(A_shared, A_local)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                    T.copy(A_shared, A_local)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_local, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_rs(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=1,
    num_threads=128,
    b_in_dtype=None,
):
    program = matmul_rs(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
        b_in_dtype,
    )

    kernel = tilelang.compile(program, out_idx=[2])
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        if in_dtype in (T.tfloat32, T.float32):
            # Convert float32 to tfloat32 because tfloat32 mma cannot truncate
            # float32 automatically, -0x1000 means
            A = (A.view(torch.int32) - 0x1000).view(torch.float32)
            B = (B.view(torch.int32) - 0x1000).view(torch.float32)
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    _b = b_in_dtype if b_in_dtype is not None else in_dtype
    if _is_fp8_dtype(in_dtype) or _is_fp8_dtype(_b):
        ins = _gen_fp8_inputs(A_shape, B_shape, in_dtype, _b)
        profiler.assert_allclose(ref_program, input_tensors=ins, atol=1e-2, rtol=1e-2)
    else:
        profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_f16f16f16_rs_tn():
    run_gemm_rs(
        512,
        1024,
        768,
        True,
        False,
        T.float16,
        T.float16,
        T.float16,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f16f16f32_rs_tn():
    run_gemm_rs(
        512,
        1024,
        768,
        True,
        False,
        T.float16,
        T.float16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f16f16f32_rs_nn():
    run_gemm_rs(
        512,
        1024,
        768,
        False,
        False,
        T.float16,
        T.float16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f16f16f32_rs_nt():
    run_gemm_rs(
        512,
        1024,
        768,
        False,
        True,
        T.float16,
        T.float16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_bf16bf16f32_rs_tn():
    run_gemm_rs(
        512,
        1024,
        768,
        True,
        False,
        T.bfloat16,
        T.bfloat16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_bf16bf16f32_rs_nn():
    run_gemm_rs(
        512,
        1024,
        768,
        False,
        False,
        T.bfloat16,
        T.bfloat16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_bf16bf16f32_rs_nt():
    run_gemm_rs(
        512,
        1024,
        768,
        False,
        True,
        T.bfloat16,
        T.bfloat16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_f16f16f16_rs_nn():
    run_gemm_rs(
        512,
        1024,
        768,
        False,
        False,
        T.float16,
        T.float16,
        T.float16,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_f16f16f16_rs_nt():
    run_gemm_rs(
        512,
        1024,
        768,
        False,
        True,
        T.float16,
        T.float16,
        T.float16,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_i8i8i32_rs_tn():
    run_gemm_rs(
        512,
        1024,
        768,
        True,
        False,
        T.int8,
        T.int8,
        T.int32,
        128,
        128,
        64,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_i8i8i32_rs_nn():
    run_gemm_rs(
        512,
        1024,
        768,
        False,
        False,
        T.int8,
        T.int8,
        T.int32,
        128,
        128,
        64,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_i8i8i32_rs_nt():
    run_gemm_rs(
        512,
        1024,
        768,
        False,
        True,
        T.int8,
        T.int8,
        T.int32,
        128,
        128,
        64,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_tf32f32f32_rs_nn():
    run_gemm_rs(
        512,
        1024,
        768,
        False,
        False,
        T.tfloat32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_tf32f32f32_rs_tn():
    run_gemm_rs(
        512,
        1024,
        768,
        True,
        False,
        T.tfloat32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_tf32f32f32_rs_nt():
    run_gemm_rs(
        512,
        1024,
        768,
        False,
        True,
        T.tfloat32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f32f32f32_rs_nn():
    run_gemm_rs(
        512,
        1024,
        768,
        False,
        False,
        T.float32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f32f32f32_rs_tn():
    run_gemm_rs(
        512,
        1024,
        768,
        True,
        False,
        T.float32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f32f32f32_rs_nt():
    run_gemm_rs(
        512,
        1024,
        768,
        False,
        True,
        T.float32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e4m3f32_rs_nn():
    run_gemm_rs(
        512,
        1024,
        768,
        False,
        False,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e4m3f32_rs_tn():
    run_gemm_rs(
        512,
        1024,
        768,
        True,
        False,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e4m3f32_rs_nt():
    run_gemm_rs(
        512,
        1024,
        768,
        False,
        True,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e5m2f32_rs_nn():
    run_gemm_rs(
        512,
        1024,
        768,
        False,
        False,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e5m2f32_rs_tn():
    run_gemm_rs(
        512,
        1024,
        768,
        True,
        False,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e5m2f32_rs_nt():
    run_gemm_rs(
        512,
        1024,
        768,
        False,
        True,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e5m2f32_rs_nn():
    run_gemm_rs(
        512,
        1024,
        768,
        False,
        False,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e5m2,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e4m3f32_rs_nn():
    run_gemm_rs(
        512,
        1024,
        768,
        False,
        False,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e4m3fn,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e5m2f32_rs_tn():
    run_gemm_rs(
        512,
        1024,
        768,
        True,
        False,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e5m2,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e5m2f32_rs_nt():
    run_gemm_rs(
        512,
        1024,
        768,
        False,
        True,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e5m2,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e4m3f32_rs_tn():
    run_gemm_rs(
        512,
        1024,
        768,
        True,
        False,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e4m3fn,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e4m3f32_rs_nt():
    run_gemm_rs(
        512,
        1024,
        768,
        False,
        True,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e4m3fn,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_f16f16f16_rs_tt():
    run_gemm_rs(
        512,
        1024,
        768,
        True,
        True,
        T.float16,
        T.float16,
        T.float16,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f16f16f32_rs_tt():
    run_gemm_rs(
        512,
        1024,
        768,
        True,
        True,
        T.float16,
        T.float16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_bf16bf16f32_rs_tt():
    run_gemm_rs(
        512,
        1024,
        768,
        True,
        True,
        T.bfloat16,
        T.bfloat16,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_i8i8i32_rs_tt():
    run_gemm_rs(
        512,
        1024,
        768,
        True,
        True,
        T.int8,
        T.int8,
        T.int32,
        128,
        128,
        64,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_tf32f32f32_rs_tt():
    run_gemm_rs(
        512,
        1024,
        768,
        True,
        True,
        T.tfloat32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
def test_ppu_gemm_f32f32f32_rs_tt():
    run_gemm_rs(
        512,
        1024,
        768,
        True,
        True,
        T.float32,
        T.float32,
        T.float32,
        128,
        128,
        32,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e4m3f32_rs_tt():
    run_gemm_rs(
        512,
        1024,
        768,
        True,
        True,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e5m2f32_rs_tt():
    run_gemm_rs(
        512,
        1024,
        768,
        True,
        True,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e4m3e5m2f32_rs_tt():
    run_gemm_rs(
        512,
        1024,
        768,
        True,
        True,
        T.float8_e4m3fn,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e5m2,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_ppu_gemm_e5m2e4m3f32_rs_tt():
    run_gemm_rs(
        512,
        1024,
        768,
        True,
        True,
        T.float8_e5m2,
        T.float32,
        T.float32,
        128,
        128,
        64,
        0,
        b_in_dtype=T.float8_e4m3fn,
    )


if __name__ == "__main__":
    tilelang.testing.main()
