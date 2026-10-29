"""PPU language dialect: common TileLang plus PPU intrinsics."""

from __future__ import annotations

from tvm import tirx

from tilelang._typing import BufferLikeType
from tilelang.language import common as _common
from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL
from tilelang.cuda.debug import device_assert  # noqa: F401
from tilelang.cuda.language.print import print as print  # noqa: F401
from tilelang.cuda.language.tir import ptx_ldmatrix, ptx_mma, ptx_mma_sp
from tilelang.language.annotations import annotate_l2_hit_ratio
from tilelang.language.builtin import (  # noqa: F401
    __ffs,
    __fns,
    __ldg,
    disable_warp_group_reg_alloc,
    get_lane_idx,
    get_warp_group_idx,
    get_warp_idx,
    get_warp_idx_sync,
    ldg128,
    ldg256,
    ldg32,
    ldg64,
    lds128,
    lds32,
    lds64,
    match_all_sync,
    match_any_sync,
    named_barrier_arrive,
    no_set_max_nreg,
    shuffle_elect,
    stg128,
    stg256,
    stg32,
    stg64,
    sts128,
    sts32,
    sts64,
)
from tilelang.ppu.intrinsics import get_swizzle_layout, make_mma_swizzle_layout  # noqa: F401
from tilelang.language.gemm_op import gemm as _dense_gemm, gemm_blockscaled as _gemm_blockscaled
from tilelang.language.math_intrinsics import (
    ieee_add,  # noqa: F401
    ieee_fdiv,  # noqa: F401
    ieee_fmaf,  # noqa: F401
    ieee_frcp,  # noqa: F401
    ieee_frsqrt,  # noqa: F401
    ieee_fsqrt,  # noqa: F401
    ieee_mul,  # noqa: F401
    ieee_sub,  # noqa: F401
)
from tilelang.language.math_intrinsics import __cos, __exp, __exp10, __log, __log2, __log10, __sin, __tan, fast_rcp
from tilelang.language.tir.ir import _dtype_forward
from tilelang.tileop.base import GemmWarpPolicy
from tilelang.utils.language import prim_expr_equal, retrieve_shape, to_buffer_region

from .copy_op import copy
from .kernel import Kernel
from .loop import Parallel, Unroll, unroll
from .reduce_op import reduce_absmax, reduce_max, reduce_min


def _validate_mxfp4_scale_regions(A, SFA, SFB, transpose_A: bool) -> None:
    """Trace-time contract for PPU MXFP4 packed E8M0 scale operands.

    Each uint16 packs two K=32 E8M0 scale bytes, so a K=64 MMA atom consumes
    exactly one scale element per row/column.
    """

    def _legalize(arg):
        if isinstance(arg, tirx.Var) and _common.has_let_value(arg):
            arg = _common.get_let_value(arg)
        return arg

    A_shape = retrieve_shape(to_buffer_region(_legalize(A)))
    K = A_shape[-2] if transpose_A else A_shape[-1]
    SFA_region = to_buffer_region(_legalize(SFA))
    SFB_region = to_buffer_region(_legalize(SFB))
    SFA_shape = retrieve_shape(SFA_region)
    SFB_shape = retrieve_shape(SFB_region)
    assert len(SFA_shape) in (1, 2) and len(SFB_shape) in (1, 2), (
        "PPU MXFP4 T.gemm scales must be one-dimensional for a K=64 tile or two-dimensional [K/64, M/N] regions for a larger K tile"
    )
    assert len(SFA_shape) == len(SFB_shape), "PPU MXFP4 scale_A and scale_B regions must have the same rank"
    if len(SFA_shape) == 1:
        assert prim_expr_equal(K, 64), (
            "One-dimensional PPU MXFP4 scales are only valid for a K=64 tile; use [K/64, M/N] scale regions for larger K tiles"
        )
    else:
        assert prim_expr_equal(SFA_shape[-2] * 64, K) and prim_expr_equal(SFB_shape[-2] * 64, K), (
            "PPU MXFP4 scale regions must contain one packed uint16 pair per K=64 MMA atom"
        )
    assert str(SFA_region.buffer.dtype) == "uint16" and str(SFB_region.buffer.dtype) == "uint16", (
        "PPU MXFP4 T.gemm scales must use uint16 packed E8M0 pairs"
    )


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
    _validate_mxfp4_scale_regions(A, SFA, SFB, transpose_A)
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
    "Parallel",
    "Unroll",
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
    "device_assert",
    "fast_rcp",
    "gemm",
    "gemm_blockscaled",
    "disable_warp_group_reg_alloc",
    "get_lane_idx",
    "get_swizzle_layout",
    "get_warp_group_idx",
    "get_warp_idx",
    "get_warp_idx_sync",
    "__ffs",
    "__fns",
    "__ldg",
    "ieee_add",
    "ieee_fdiv",
    "ieee_fmaf",
    "ieee_frcp",
    "ieee_frsqrt",
    "ieee_fsqrt",
    "ieee_mul",
    "ieee_sub",
    "ldg128",
    "ldg256",
    "ldg32",
    "ldg64",
    "lds128",
    "lds32",
    "lds64",
    "make_mma_swizzle_layout",
    "match_all_sync",
    "match_any_sync",
    "named_barrier_arrive",
    "no_set_max_nreg",
    "pdl_sync",
    "pdl_trigger",
    "print",
    "ppu_aiu_load",
    "ptx_ldmatrix",
    "ptx_mma",
    "ptx_mma_scaled",
    "ptx_mma_sp",
    "reduce_absmax",
    "reduce_max",
    "reduce_min",
    "rng_init",
    "rng_rand",
    "rng_rand_float",
    "shuffle_elect",
    "stg128",
    "stg256",
    "stg32",
    "stg64",
    "sts128",
    "sts32",
    "sts64",
    "tix_ldmatrix_swzl",
    "unroll",
)

del _COMMON_ALL
