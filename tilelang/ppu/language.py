"""PPU language dialect: common TileLang plus PPU intrinsics."""

from __future__ import annotations

from tvm import tirx

from tilelang.language import common as _common
from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL
from tilelang.language.annotations import annotate_l2_hit_ratio
from tilelang.language.builtin import disable_warp_group_reg_alloc, shuffle_elect
from tilelang.language.tir import op as _tir_op
from tilelang.language.tir.ir import _dtype_forward

ptx_ldmatrix = _dtype_forward(_tir_op.ptx_ldmatrix)
ptx_mma = _dtype_forward(_tir_op.ptx_mma)
ptx_mma_sp = _dtype_forward(_tir_op.ptx_mma_sp)


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
    "annotate_l2_hit_ratio",
    "disable_warp_group_reg_alloc",
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
