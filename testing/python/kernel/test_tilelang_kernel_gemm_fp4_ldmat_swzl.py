"""Test FP4 GEMM with the swizzled ldmatrix (ldmat.swzl) on PPU1.5.

FP4 (e2m1) GEMM in the TN form:

  * A [M, K] non-transposed  -> non-trans swizzle instruction
  * B [N, K] transposed       -> non-trans swizzle instruction (with
                                 trans_block register shuffle)

Both shared operands use the Full/Half bank swizzle layout, and the load
is emitted as ``tl::tix_ldmatrix_swzl_x4``. The swizzle mode is decided
by the shared row width measured in bytes:

  * 128B (block_K=256 for fp4, 256*4/8=128) -> swzl_mode=0
  *  64B (block_K=128 for fp4, 128*4/8=64)  -> swzl_mode=1
"""

import re

import torch

import tilelang
import tilelang.ppu.language as T
import tilelang.testing
from tilelang.transform import PassConfigKey

# Global matrix dimensions
M = N = K = 256

# e2m1 magnitude LUT; bit 3 = sign.
_FP4_E2M1_MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _float_to_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Quantize a float tensor to e2m1 codes (uint8 in [0, 15])."""
    x = x.clamp(-6.0, 6.0)
    sign = (x < 0).to(torch.uint8) << 3
    lut = torch.tensor(_FP4_E2M1_MAG, device=x.device, dtype=torch.float32)
    mag = (x.abs().unsqueeze(-1) - lut).abs().argmin(dim=-1).to(torch.uint8)
    return sign | mag


def _pack_fp4(codes: torch.Tensor) -> torch.Tensor:
    """Pack e2m1 codes (..., K) into int8 (..., K // 2); even idx -> low nibble."""
    lo = codes[..., 0::2]
    hi = codes[..., 1::2]
    return (lo | (hi << 4)).view(torch.int8)


def _dequant_fp4(packed: torch.Tensor) -> torch.Tensor:
    """int8 (..., K // 2) -> fp32 (..., K)."""
    u = packed.view(torch.uint8)
    lo = (u & 0x0F).to(torch.int64)
    hi = (u >> 4).to(torch.int64)
    codes = torch.stack([lo, hi], dim=-1).flatten(-2)
    mag_lut = torch.tensor(_FP4_E2M1_MAG, device=u.device, dtype=torch.float32)
    mag = mag_lut[codes & 0x7]
    sign = torch.where((codes & 0x8) != 0, -1.0, 1.0)
    return mag * sign


def _gen_fp4_inputs(A_shape, B_shape):
    """Quantize random fp16 to e2m1 on CPU, pack into int8, move to device."""
    a = torch.randn(*A_shape, dtype=torch.float16)
    b = torch.randn(*B_shape, dtype=torch.float16)
    a_packed = _pack_fp4(_float_to_e2m1_codes(a.float()))
    b_packed = _pack_fp4(_float_to_e2m1_codes(b.float()))
    return a_packed.to("cuda"), b_packed.to("cuda")


def gemm_fp4_kernel(block_M, block_N, block_K, dtype=T.float4_e2m1fn, accum_dtype=T.float32):
    """Construct an FP4 TN-form GEMM prim_func with the given tile sizes."""

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
    """Compile the FP4 kernel with ldmat.swzl enabled and verify numerics."""
    kernel = tilelang.compile(
        gemm_fp4_kernel(block_M, block_N, block_K),
        out_idx=[-1],
        pass_configs={
            PassConfigKey.TL_DISABLE_AIU_LOWER: False,
            PassConfigKey.TL_DISABLE_LDMAT_SWZL: False,
        },
    )

    # FP4 inputs: A [M, K], B [N, K] in TN form
    a_packed, b_packed = _gen_fp4_inputs((M, K), (N, K))
    c = kernel(a_packed, b_packed)

    # Reference: dequantize on CPU and do float32 matmul
    a_f32 = _dequant_fp4(a_packed.cpu())
    b_f32 = _dequant_fp4(b_packed.cpu())
    ref_c = (a_f32 @ b_f32.T).to(c.device)

    source = kernel.get_kernel_source()
    return c, ref_c, source


def _swzl_calls(source: str):
    """Extract all non-trans swizzled ldmatrix calls from the generated source."""
    return re.findall(r"tl::tix_ldmatrix_swzl_x4\([^;]+\)", source)


# The trailing trans_block flag may be printed as (bool)0/(bool)1 or true/false.
_TAIL_RE = r", (0|1), (?:\(bool\))?(?:true|false|0|1)\)"


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_fp4_swzl_64b():
    """64B swizzle: block_K=128 fp4 -> 128*4/8=64 bytes per row -> swzl_mode=1."""
    c, ref_c, source = compile_and_verify(block_M=128, block_N=128, block_K=128)

    torch.testing.assert_close(c, ref_c, rtol=2e-2, atol=2e-2)

    calls = _swzl_calls(source)
    assert len(calls) > 0, "Expected tl::tix_ldmatrix_swzl_x4 in generated CUDA source"
    assert "tix_ldmatrix_swzl_x4_trans" not in source, "FP4 swizzle should only use the non-trans swizzled ldmatrix"
    for call in calls:
        # mode 1: the mode argument is 1 (before the trans_block flag)
        m = re.search(_TAIL_RE, call)
        assert m is not None and m.group(1) == "1", f"Expected swzl_mode=1 for 64B swizzle, got: {call}"


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_fp4_swzl_128b():
    """128B swizzle: block_K=256 fp4 -> 256*4/8=128 bytes per row -> swzl_mode=0."""
    c, ref_c, source = compile_and_verify(block_M=64, block_N=64, block_K=256)

    torch.testing.assert_close(c, ref_c, rtol=2e-2, atol=2e-2)

    calls = _swzl_calls(source)
    assert len(calls) > 0, "Expected tl::tix_ldmatrix_swzl_x4 in generated CUDA source"
    assert "tix_ldmatrix_swzl_x4_trans" not in source, "FP4 swizzle should only use the non-trans swizzled ldmatrix"
    for call in calls:
        # mode 0: the mode argument is 0 (before the trans_block flag)
        m = re.search(_TAIL_RE, call)
        assert m is not None and m.group(1) == "0", f"Expected swzl_mode=0 for 128B swizzle, got: {call}"


if __name__ == "__main__":
    tilelang.testing.main()
