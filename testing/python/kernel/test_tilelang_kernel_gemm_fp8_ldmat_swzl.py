"""Test FP8 GEMM with the swizzled ldmatrix (ldmat.swzl) on PPU1.5.

FP8 (e4m3) GEMM in the TN form:

  * A [M, K] non-transposed  -> non-trans swizzle instruction
  * B [N, K] transposed       -> non-trans swizzle instruction (with
                                 trans_block register shuffle)

Both shared operands use the Full/Half bank swizzle layout, and the load
is emitted as ``tl::tix_ldmatrix_swzl_x4``. The swizzle mode is decided
by the shared row width measured in bytes:

  * 128B (block_K=128 for fp8) -> swzl_mode=0
  * 64B  (block_K=64  for fp8) -> swzl_mode=1
"""

import re

import torch

import tilelang
import tilelang.ppu.language as T
import tilelang.testing
from tilelang.transform import PassConfigKey

# Global matrix dimensions
M = N = K = 256


def gemm_fp8_kernel(block_M, block_N, block_K, dtype=T.float8_e4m3, accum_dtype=T.float32):
    """Construct an FP8 TN-form GEMM prim_func with the given tile sizes."""

    @T.prim_func
    def main(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((N, K), dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_N, block_K), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)

            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def compile_and_verify(block_M, block_N, block_K):
    """Compile the FP8 kernel with ldmat.swzl enabled and verify numerics."""
    kernel = tilelang.compile(
        gemm_fp8_kernel(block_M, block_N, block_K),
        out_idx=[-1],
        pass_configs={
            PassConfigKey.TL_DISABLE_AIU_LOWER: False,
            PassConfigKey.TL_DISABLE_LDMAT_SWZL: False,
        },
    )

    a = torch.randn(M, K, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
    b = torch.randn(N, K, dtype=torch.float16, device="cuda").to(torch.float8_e4m3fn)
    c = kernel(a, b)
    ref_c = a.to(torch.float32) @ b.to(torch.float32).t()

    source = kernel.get_kernel_source()
    return c, ref_c, source


def _swzl_calls(source: str):
    """Extract all non-trans swizzled ldmatrix calls from the generated source."""
    return re.findall(r"tl::tix_ldmatrix_swzl_x4\([^;]+\)", source)


# The trailing trans_block flag may be printed as (bool)0/(bool)1 or true/false.
_TAIL_RE = r", (0|1), (?:\(bool\))?(?:true|false|0|1)\)"


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_fp8_swzl_128b():
    """128B swizzle: block_K=128 fp8 -> 128 bytes per row -> swzl_mode=0."""
    c, ref_c, source = compile_and_verify(block_M=128, block_N=128, block_K=128)

    torch.testing.assert_close(c, ref_c, rtol=2e-2, atol=2e-2)

    calls = _swzl_calls(source)
    assert len(calls) > 0, "Expected tl::tix_ldmatrix_swzl_x4 in generated CUDA source"
    assert "tix_ldmatrix_swzl_x4_trans" not in source, "FP8 swizzle should only use the non-trans swizzled ldmatrix"
    for call in calls:
        # mode 0: the mode argument is 0 (before the trans_block flag)
        m = re.search(_TAIL_RE, call)
        assert m is not None and m.group(1) == "0", f"Expected swzl_mode=0 for 128B swizzle, got: {call}"


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_fp8_swzl_64b():
    """64B swizzle: block_K=64 fp8 -> 64 bytes per row -> swzl_mode=1."""
    c, ref_c, source = compile_and_verify(block_M=128, block_N=128, block_K=64)

    torch.testing.assert_close(c, ref_c, rtol=2e-2, atol=2e-2)

    calls = _swzl_calls(source)
    assert len(calls) > 0, "Expected tl::tix_ldmatrix_swzl_x4 in generated CUDA source"
    assert "tix_ldmatrix_swzl_x4_trans" not in source, "FP8 swizzle should only use the non-trans swizzled ldmatrix"
    for call in calls:
        # mode 1: the mode argument is 1 (before the trans_block flag)
        m = re.search(_TAIL_RE, call)
        assert m is not None and m.group(1) == "1", f"Expected swzl_mode=1 for 64B swizzle, got: {call}"


if __name__ == "__main__":
    tilelang.testing.main()
