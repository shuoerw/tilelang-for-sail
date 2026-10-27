"""PPU language dialect: common TileLang plus PPU intrinsics."""

from __future__ import annotations

from tvm import tirx

from tilelang._typing import BufferLikeType
from tilelang.language import common as _common
from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL
from tilelang.cuda.language.tir import ptx_ldmatrix, ptx_mma, ptx_mma_sp
from tilelang.language.annotations import annotate_l2_hit_ratio
from tilelang.language.builtin import disable_warp_group_reg_alloc, get_lane_idx, no_set_max_nreg, shuffle_elect
from tilelang.language.gemm_op import gemm as _dense_gemm, gemm_blockscaled as _gemm_blockscaled
from tilelang.language.math_intrinsics import __cos, __exp, __exp10, __log, __log2, __log10, __sin, __tan, fast_rcp
from tilelang.language.tir.ir import _dtype_forward
from tilelang.tileop.base import GemmWarpPolicy

from .copy_op import copy
from .kernel import Kernel


def gemm_blockscaled(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    SFA: BufferLikeType,
    SFB: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    *,
    k_start: int | tirx.PrimExpr,
    sf_a_granularity_k: int,
    sf_b_granularity_k: int,
    annotations: dict | None = None,
) -> tirx.PrimExpr:
    """PPU MXFP4 GEMM with packed E8M0 pairs covering two K32 blocks."""
    if sf_a_granularity_k != 32 or sf_b_granularity_k != 32:
        raise ValueError("PPU MXFP4 block scales require sf_a_granularity_k=sf_b_granularity_k=32")
    return _gemm_blockscaled(
        A,
        B,
        C,
        SFA,
        SFB,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        k_start=k_start,
        sf_a_granularity_k=sf_a_granularity_k,
        sf_b_granularity_k=sf_b_granularity_k,
        annotations=annotations,
    )


def gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
    annotations: dict | None = None,
    *,
    scale_A: BufferLikeType | None = None,
    scale_B: BufferLikeType | None = None,
) -> tirx.PrimExpr:
    """PPU GEMM with backward-compatible packed MXFP4 scale operands."""
    if (scale_A is None) != (scale_B is None):
        raise ValueError("scale_A and scale_B must be provided together")
    if scale_A is None:
        return _dense_gemm(
            A,
            B,
            C,
            transpose_A,
            transpose_B,
            policy,
            clear_accum,
            annotations,
        )
    return gemm_blockscaled(
        A,
        B,
        C,
        scale_A,
        scale_B,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        k_start=0,
        sf_a_granularity_k=32,
        sf_b_granularity_k=32,
        annotations=annotations,
    )


def tix_ldmatrix_swzl(trans, num, src_access_ptr, dst_access_ptr, swzl_mode, trans_block=False):
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tix_ldmatrix_swzl"),
        trans,
        num,
        src_access_ptr,
        dst_access_ptr,
        swzl_mode,
        trans_block,
    )


def ppu_aiu_load(smem_ptr, gmem_ptr, dim_c, dim_w, cube_c, cube_w, stride_w_bytes, start_c, start_w, swzl_mode, is_swizzle=1):
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.ppu_aiu_load"),
        smem_ptr,
        gmem_ptr,
        dim_c,
        dim_w,
        cube_c,
        cube_w,
        stride_w_bytes,
        start_c,
        start_w,
        swzl_mode,
        is_swizzle,
        annotations={"no_implicit_async_commit_wait": tirx.const(1, "int32")},
    )


@_dtype_forward
def ptx_mma_scaled(
    dtype,
    shape,
    A_layout,
    B_layout,
    A_dtype,
    B_dtype,
    C_dtype,
    multiplicand_a,
    a_index,
    multiplicand_b,
    b_index,
    accumulator,
    c_index,
    saturate,
    scale_a,
    scale_b,
    scale_a_selector,
    scale_b_selector,
):
    return tirx.call_intrin(
        dtype,
        "tirx.ptx_mma",
        shape,
        A_layout,
        B_layout,
        A_dtype,
        B_dtype,
        C_dtype,
        multiplicand_a,
        a_index,
        multiplicand_b,
        b_index,
        accumulator,
        c_index,
        saturate,
        scale_a,
        scale_b,
        scale_a_selector,
        scale_b_selector,
    )


def rng_init(seed, seq=None, off=0, generator="acrandStatePhilox4_32_10_t"):
    if generator not in {"acrandStateMRG32k3a_t", "acrandStatePhilox4_32_10_t", "acrandStateXORWOW_t"}:
        raise ValueError(f"Unsupported PPU random generator: {generator}")
    if seq is None:
        seq = _common.get_thread_binding() + _common.get_block_binding() * _common.get_thread_extent()
    return tirx.call_intrin("void", tirx.op.Op.get("tl.rng_init"), tirx.convert(seed), tirx.convert(seq), tirx.convert(off), generator)


def rng_rand():
    return tirx.call_intrin("uint32", tirx.op.Op.get("tl.rng_rand"))


def rng_rand_float(bit=32, dist="uniform"):
    if bit not in (32, 64) or dist not in ("uniform", "normal"):
        raise ValueError(f"Unsupported PPU random distribution: bit={bit}, dist={dist}")
    return tirx.call_intrin(f"float{bit}", tirx.op.Op.get("tl.rng_rand_float"), dist)


def pdl_trigger():
    return tirx.call_intrin("void", tirx.op.Op.get("tl.pdl_trigger"))


def pdl_sync():
    return tirx.call_intrin("void", tirx.op.Op.get("tl.pdl_sync"))


__tilelang_dialect__ = "ppu"
__all__ = (
    *_COMMON_ALL,
    "Kernel",
    "__cos",
    "__exp",
    "__exp10",
    "__log",
    "__log2",
    "__log10",
    "__sin",
    "__tan",
    "annotate_l2_hit_ratio",
    "copy",
    "fast_rcp",
    "gemm",
    "gemm_blockscaled",
    "disable_warp_group_reg_alloc",
    "get_lane_idx",
    "no_set_max_nreg",
    "pdl_sync",
    "pdl_trigger",
    "ppu_aiu_load",
    "ptx_ldmatrix",
    "ptx_mma",
    "ptx_mma_scaled",
    "ptx_mma_sp",
    "rng_init",
    "rng_rand",
    "rng_rand_float",
    "shuffle_elect",
    "tix_ldmatrix_swzl",
)

del _COMMON_ALL
