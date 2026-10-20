"""PPU-specific low-level TIR language operators."""

from __future__ import annotations

import builtins

import tvm.tirx.op as _tvm_op
from tvm import tirx

from tilelang.language.tir.ir import _dtype_forward

__all__ = [
    "ppu_aiu_load",
    "ptx_ldmatrix",
    "ptx_mma",
    "ptx_mma_scaled",
    "ptx_mma_sp",
    "tix_ldmatrix_swzl",
]

# Unchanged upstream PTX operators are rebound directly in the PPU dialect.
ptx_mma = _dtype_forward(_tvm_op.ptx_mma)


def ptx_ldmatrix(trans, num, src_access_ptr, dst_access_ptr):
    """TileLang intrinsic for ptx load matrix from shared memory

    Uses `tl.ptx_ldmatrix` which expects access pointers created via
    `T.access_ptr` (i.e. `tl.access_ptr` wrapping a `BufferLoad`).

    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-ldmatrix

    Parameters
    ----------
    trans : bool
        The matrix is loaded in column-major format.

    num : IntImm
        The number of matrices (2 or 4).

    src_access_ptr : PrimExpr
        A `tl.access_ptr` pointing to the source (shared memory) buffer.

    dst_access_ptr : PrimExpr
        A `tl.access_ptr` pointing to the destination (local/register) buffer.

    Returns
    -------
    call : PrimExpr
        The call expression (handle-typed).
    """
    return tirx.call_intrin(
        "handle",
        tirx.op.Op.get("tl.ptx_ldmatrix"),
        trans,
        num,
        src_access_ptr,
        dst_access_ptr,
    )


@_dtype_forward
def ptx_mma_sp(
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
    metadata,
    meta_index,
    sparse_selector,
    saturate,
):
    """TVM intrinsic for sparse tensor core ptx instructions
    https://docs.nvidia.com/cuda/parallel-thread-execution/index.html#warp-level-matrix-instructions-for-sparse-mma

    Parameters
    ----------
    dtype : str
        The data type of the result.

    shape : str
        The shape of mma fragment.

    A_layout : Literal["row", "col"]
        The layout of multiplicand fragment A.

    B_layout : Literal["row", "col"]
        The layout of multiplicand fragment B.

    A_dtype : str
        The data type of multiplicand fragment A.

    B_dtype : str
        The data type of multiplicand fragment B.

    C_dtype : str
        The data type of accumulator fragment C.

    multiplicand_a : Var
        The multiplicand fragment A variable.

    a_index : Expr
        The index of multiplicand fragment A.

    multiplicand_b : Var
        The multiplicand fragment B variable.

    b_index : Expr
        The index of multiplicand fragment B.

    accumulator : Var
        The accumulator fragment C variable.

    c_index : Expr
        The index of accumulator fragment C.

    metadata : Expr
        The metadata of operand.

    meta_index : Expr
        The metadata index of operand.

    sparse_selector : Expr
        The sparse selector indicating the thread that stores the metadata.

    saturate : bool
        The optional saturation at the output.

    Returns
    -------
    call : PrimExpr
        The call expression.
    """
    return _tvm_op.ptx_mma_sp(
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
        metadata,
        meta_index,
        sparse_selector,
        saturate,
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


def ppu_aiu_load(
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
    is_swizzle=1,
    *,
    prefetch_bytes: int = 128,
):
    """TIX AIU Load operation.

    Directly invokes the backend ppu_aiu_load with a per-load LLC prefetch hint.

    Parameters
    ----------
    smem_ptr : PrimExpr
        Shared memory destination pointer.
    gmem_ptr : PrimExpr
        Global memory source pointer.
    dim_c : int | PrimExpr
        C dimension size (bytes).
    dim_w : int | PrimExpr
        W dimension size (rows).
    cube_c : int
        Copy block C size (bytes).
    cube_w : int
        Copy block W size (rows).
    stride_w_bytes : int | PrimExpr
        Row stride in bytes.
    start_c : int | PrimExpr
        Start coordinate on C axis (bytes).
    start_w : int | PrimExpr
        Start coordinate on W axis (rows).
    swzl_mode : int
        Swizzle mode (0=128B, 1=64B).
    is_swizzle : bool | int, optional
        Use a swizzled destination (1, default) or a linear destination (0).
    prefetch_bytes : int, optional
        Compile-time LLC prefetch size: 64, 128 (default), or 256 bytes.
        Zero omits the prefetch hint. This does not change the copy extent.
    """

    if isinstance(prefetch_bytes, tirx.IntImm):
        prefetch_bytes = prefetch_bytes.value
    if isinstance(prefetch_bytes, builtins.bool) or not isinstance(prefetch_bytes, builtins.int):
        raise TypeError("ppu_aiu_load prefetch_bytes must be a compile-time integer")
    if prefetch_bytes not in (0, 64, 128, 256):
        raise ValueError("ppu_aiu_load prefetch_bytes must be 0, 64, 128, or 256")

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
        prefetch_bytes,
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
