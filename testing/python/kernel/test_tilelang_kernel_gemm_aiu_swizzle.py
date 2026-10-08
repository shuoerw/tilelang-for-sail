"""Test AIU load swizzle modes in GEMM kernels.

Verifies that TileLang correctly generates `tl::aiu_load` instructions with
the appropriate swizzle mode parameter for different shared memory layouts:

  * 128B swizzle (FullBankSwizzle): swzl_mode=0, when continuous_dim % 64 == 0
  * 64B swizzle (HalfBankSwizzle): swzl_mode=1, when continuous_dim % 32 == 0

Also tests the swizzle_row_period constraint: when outer_per_warp < 8 (the
shared period for both 64B and 128B swizzle), the number of warps participating
in AIU load is reduced to satisfy alignment.

Each test compiles the kernel, verifies numerical correctness against torch
matmul, and inspects the generated CUDA source for expected aiu_load patterns.
"""

import torch
import tilelang
import tilelang.ppu.language as T
import tilelang.testing
from tilelang.transform import PassConfigKey

# Global matrix dimensions
M = N = K = 512


def gemm_kernel(block_M, block_N, block_K, dtype=T.float16, accum_dtype=T.float32):
    """Construct a GEMM prim_func with the given tile dimensions."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
                T.copy(A[by * block_M, k * block_K], A_shared, prefer_instruction="aiu")
                T.copy(B[k * block_K, bx * block_N], B_shared, prefer_instruction="aiu")
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def compile_and_verify(block_M, block_N, block_K):
    """Compile a GEMM kernel with AIU enabled, run it, and return (result, ref, source)."""
    kernel = tilelang.compile(
        gemm_kernel(block_M, block_N, block_K),
        out_idx=[-1],
        pass_configs={
            PassConfigKey.TL_DISABLE_AIU_LOWER: False,
        },
    )

    a = torch.randn(M, K, device="cuda", dtype=torch.float16)
    b = torch.randn(K, N, device="cuda", dtype=torch.float16)
    c = kernel(a, b)
    ref_c = a @ b
    source = kernel.get_kernel_source()

    return c, ref_c, source


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_128b_swizzle():
    """128B swizzle: block_M=128, block_K=64.

    A_shared (128, 64): continuous=64, 64 % 64 == 0 → 128B swizzle
    B_shared (64, 128): continuous=128, 128 % 64 == 0 → 128B swizzle
    All aiu_load calls should have swzl_mode=0.
    """
    c, ref_c, source = compile_and_verify(block_M=128, block_N=128, block_K=64)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    assert "tl::aiu_load" in source, "Expected tl::aiu_load in generated CUDA source"

    # 128B swizzle → swzl_mode=0: all aiu_load should have swzl_mode=0 (last arg)
    aiu_load_lines = [line for line in source.splitlines() if "tl::aiu_load(" in line]
    assert len(aiu_load_lines) > 0
    for line in aiu_load_lines:
        args = line.split("tl::aiu_load(", 1)[1].split(");", 1)[0].split(",")
        assert args[-1].strip() == "0", f"Expected swzl_mode=0 for 128B swizzle, got: {line}"


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_64b_swizzle():
    """64B swizzle for A, 128B for B: block_M=128, block_K=32.

    A_shared (128, 32): continuous=32, 32 % 32 == 0 → 64B swizzle (swzl_mode=1)
    B_shared (32, 128): continuous=128, 128 % 64 == 0 → 128B swizzle (swzl_mode=0)
    Source should contain at least one aiu_load with swzl_mode=1.
    """
    c, ref_c, source = compile_and_verify(block_M=128, block_N=128, block_K=32)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    assert "tl::aiu_load" in source, "Expected tl::aiu_load in generated CUDA source"

    # Should have at least one aiu_load with swzl_mode=1 (for A's 64B swizzle)
    aiu_load_lines = [line for line in source.splitlines() if "tl::aiu_load(" in line]
    assert len(aiu_load_lines) > 0
    has_mode_1 = any(line.split("tl::aiu_load(", 1)[1].split(");", 1)[0].split(",")[-1].strip() == "1" for line in aiu_load_lines)
    assert has_mode_1, "Expected at least one aiu_load with swzl_mode=1 for 64B swizzle on A_shared"


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_64b_swizzle_constraint_trigger():
    """64B swizzle constraint trigger: block_M=16, block_K=32.

    A_shared (16, 32): 64B swizzle, outer_dim=16, initial outer_per_warp=4 < 8
    → constraint triggers, max_outer_splits=2, outer_per_warp raised to 8.
    B_shared (32, 64): 128B swizzle, outer_per_warp=8 (just satisfies).
    """
    c, ref_c, source = compile_and_verify(block_M=16, block_N=64, block_K=32)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    assert "tl::aiu_load" in source, "Expected tl::aiu_load in generated CUDA source"

    # Should contain aiu_load with swzl_mode=1 (64B for A)
    aiu_load_lines = [line for line in source.splitlines() if "tl::aiu_load(" in line]
    assert len(aiu_load_lines) > 0
    has_mode_1 = any(line.split("tl::aiu_load(", 1)[1].split(");", 1)[0].split(",")[-1].strip() == "1" for line in aiu_load_lines)
    assert has_mode_1, "Expected aiu_load with swzl_mode=1 for 64B swizzle on A_shared"


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_128b_swizzle_constraint_trigger():
    """128B swizzle constraint trigger: block_M=16, block_K=64.

    A_shared (16, 64): 128B swizzle, outer_dim=16, initial outer_per_warp=4 < 8
    → constraint triggers, max_outer_splits=2, outer_per_warp raised to 8.
    Only 2 warps participate in A's AIU load.
    """
    c, ref_c, source = compile_and_verify(block_M=16, block_N=128, block_K=64)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    assert "tl::aiu_load" in source, "Expected tl::aiu_load in generated CUDA source"

    # 128B swizzle → swzl_mode=0
    aiu_load_lines = [line for line in source.splitlines() if "tl::aiu_load(" in line]
    assert len(aiu_load_lines) > 0
    for line in aiu_load_lines:
        args = line.split("tl::aiu_load(", 1)[1].split(");", 1)[0].split(",")
        assert args[-1].strip() == "0", f"Expected swzl_mode=0 for 128B swizzle, got: {line}"


def fp8_aiu_gemm_kernel(block_k, dtype):
    """Construct an explicit FP8 AIU GEMM for one swizzle width."""

    @T.prim_func
    def main(
        A: T.Tensor((128, block_k), dtype),
        B: T.Tensor((128, block_k), dtype),
        C: T.Tensor((128, 128), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((128, block_k), dtype)
            B_shared = T.alloc_shared((128, block_k), dtype)
            C_local = T.alloc_fragment((128, 128), T.float32)

            T.clear(C_local)
            T.copy(A, A_shared, prefer_instruction="aiu")
            T.copy(B, B_shared, prefer_instruction="aiu")
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C)

    return main


def compile_and_verify_fp8_aiu(block_k, expected_swizzle_mode, tilelang_dtype, torch_dtype):
    kernel = tilelang.compile(
        fp8_aiu_gemm_kernel(block_k, tilelang_dtype),
        out_idx=[-1],
        pass_configs={
            PassConfigKey.TL_DISABLE_AIU_LOWER: False,
            PassConfigKey.TL_DISABLE_LDMAT_SWZL: False,
        },
    )
    source = kernel.get_kernel_source()
    aiu_load_calls = [line for line in source.splitlines() if "tl::aiu_load(" in line]
    assert aiu_load_calls, "Expected tl::aiu_load in generated source"
    for call in aiu_load_calls:
        args = call.split("tl::aiu_load(", 1)[1].split(");", 1)[0].split(",")
        assert args[-1].strip() == str(expected_swizzle_mode), call

    a = torch.randn((128, block_k), device="cuda", dtype=torch.float16).to(torch_dtype)
    b = torch.randn((128, block_k), device="cuda", dtype=torch.float16).to(torch_dtype)
    out = kernel(a, b)
    ref = a.float() @ b.float().T
    rel_l2 = torch.linalg.vector_norm(out - ref) / torch.linalg.vector_norm(ref)
    assert float(rel_l2) <= 0.02, float(rel_l2)


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_fp8_e4m3fn_64b_swizzle():
    compile_and_verify_fp8_aiu(
        block_k=64,
        expected_swizzle_mode=1,
        tilelang_dtype=T.float8_e4m3fn,
        torch_dtype=torch.float8_e4m3fn,
    )


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_fp8_e4m3fn_128b_swizzle():
    compile_and_verify_fp8_aiu(
        block_k=128,
        expected_swizzle_mode=0,
        tilelang_dtype=T.float8_e4m3fn,
        torch_dtype=torch.float8_e4m3fn,
    )


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_fp8_e5m2_64b_swizzle():
    compile_and_verify_fp8_aiu(
        block_k=64,
        expected_swizzle_mode=1,
        tilelang_dtype=T.float8_e5m2,
        torch_dtype=torch.float8_e5m2,
    )


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_fp8_e5m2_128b_swizzle():
    compile_and_verify_fp8_aiu(
        block_k=128,
        expected_swizzle_mode=0,
        tilelang_dtype=T.float8_e5m2,
        torch_dtype=torch.float8_e5m2,
    )


if __name__ == "__main__":
    tilelang.testing.main()
