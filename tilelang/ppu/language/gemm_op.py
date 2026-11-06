"""PPU dialect of the GEMM operators: common GEMM plus PPU MXFP4 scale rules."""

from __future__ import annotations

from tvm import tirx

from tilelang._typing import BufferLikeType
from tilelang.language import common as _common
from tilelang.language.gemm_op import gemm as _dense_gemm, gemm_blockscaled as _gemm_blockscaled
from tilelang.tileop.base import GemmWarpPolicy
from tilelang.utils.language import prim_expr_equal, retrieve_shape, to_buffer_region

__all__ = ["gemm", "gemm_blockscaled"]


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
    """PPU GEMM with packed MXFP4 scales and automatic register pipelining.

    Eligible inner GEMM loops use a two-slot register pipeline automatically.
    """
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
