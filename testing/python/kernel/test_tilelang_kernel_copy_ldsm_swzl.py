"""Test SWZL x4 shared-to-fragment copies on PPU0015.

SWZL loads require a full- or half-bank shared layout and a fragment
mapping that matches a 2x2 packet of 8-row matrices. Incompatible packet
shapes use per-row matrix loads. Coverage includes both register orders,
b16 transposition, slices, packed b8 copies, and GEMM-inferred layouts.
"""

import re

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import (
    make_full_bank_swizzled_layout,
    make_gemm_fragment_8x8,
    make_gemm_fragment_8x8_transposed,
    make_half_bank_swizzled_layout,
    make_linear_layout,
    make_quarter_bank_swizzled_layout,
    make_swizzled_layout,
)

SWZL = "tl::tix_ldmatrix_swzl_x4"
SWZL_TRANS = "tl::tix_ldmatrix_swzl_x4_trans"


def _frag_8x16(rep_rows, rep_cols):
    base = T.Fragment((8, 16), forward_fn=lambda i, j: (j // 4 + 4 * i, j % 4))
    return base.repeat([rep_rows, rep_cols], repeat_on_thread=False)


def _make_raw_kernel(dtype, shared_shape, frag_shape, swizzle_fn, sliced=None, row_major=True, transposed=False):
    """Raw S->R copy out of a swizzled shared tensor; sliced=(row_start, col_start)."""
    base = make_gemm_fragment_8x8_transposed() if transposed else make_gemm_fragment_8x8()
    frag_layout = base.repeat([frag_shape[0] // 8, frag_shape[1] // 8], repeat_on_thread=False, lower_dim_first=row_major)
    row_start, col_start = sliced if sliced is not None else (0, 0)

    @T.prim_func
    def main(A: T.Tensor(shared_shape, dtype), B: T.Tensor(frag_shape, dtype)):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared(shared_shape, dtype)
            fragment = T.alloc_fragment(frag_shape, dtype)
            T.annotate_layout({shared: swizzle_fn(shared), fragment: frag_layout})
            T.copy(A, shared)
            T.copy(shared[row_start : row_start + frag_shape[0], col_start : col_start + frag_shape[1]], fragment)
            T.copy(fragment, B)

    return main


def _run_raw_case(dtype, shared_shape, frag_shape, swizzle_fn, mode, sliced=None, row_major=True):
    kernel = tilelang.compile(_make_raw_kernel(dtype, shared_shape, frag_shape, swizzle_fn, sliced, row_major), out_idx=[-1])
    source = kernel.get_kernel_source()
    calls = re.findall(re.escape(SWZL) + r"\([^;]+;", source)
    assert len(calls) > 0, f"Expected {SWZL} in generated CUDA source"
    flag = "1" if row_major else "0"
    for call in calls:
        assert call.rstrip().endswith(f", {mode}, {flag});"), f"Expected swzl_mode={mode} trans_block={flag}, got: {call}"

    generator = torch.Generator(device="cuda").manual_seed(0)
    values = torch.randint(-128, 128, shared_shape, generator=generator, device="cuda", dtype=torch.int32)
    if dtype == "int8":
        data = values.to(torch.int8)
    elif dtype == "float8_e4m3fn":
        data = values.to(torch.int8).view(torch.float8_e4m3fn)
    else:
        data = values.to(torch.bfloat16)
    out = kernel(data)
    row_start, col_start = sliced or (0, 0)
    ref = data[row_start : row_start + frag_shape[0], col_start : col_start + frag_shape[1]]
    if dtype == "float8_e4m3fn":
        torch.testing.assert_close(out.view(torch.uint8), ref.view(torch.uint8), rtol=0, atol=0)
    else:
        torch.testing.assert_close(out, ref, rtol=0, atol=0)


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_swzl_b16_full_bank_both_directions():
    """bf16, 128B swizzle (mode 0): row-major (trans_block=1) and K-major (0)."""
    _run_raw_case("bfloat16", (16, 64), (16, 16), make_full_bank_swizzled_layout, 0, row_major=True)
    _run_raw_case("bfloat16", (16, 64), (16, 16), make_full_bank_swizzled_layout, 0, row_major=False)


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_swzl_b16_trans_full_bank():
    """bf16 transposed fragment, 128B swizzle (mode 0), uses the trans helper."""
    shape = (16, 64)
    frag_shape = (16, 16)
    frag_layout = make_gemm_fragment_8x8_transposed().repeat(
        [frag_shape[0] // 8, frag_shape[1] // 8], repeat_on_thread=False
    )

    @T.prim_func
    def main(A: T.Tensor(shape, "bfloat16"), B: T.Tensor(frag_shape, "bfloat16")):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared(shape, "bfloat16")
            fragment = T.alloc_fragment(frag_shape, "bfloat16")
            T.annotate_layout({shared: make_full_bank_swizzled_layout(shared), fragment: frag_layout})
            T.copy(A, shared)
            T.copy(shared[0:frag_shape[0], 0:frag_shape[1]], fragment)
            T.copy(fragment, B)

    kernel = tilelang.compile(main, out_idx=[-1])
    source = kernel.get_kernel_source()
    calls = re.findall(re.escape(SWZL_TRANS) + r"\([^;]+;", source)
    assert len(calls) > 0, f"Expected {SWZL_TRANS} in generated CUDA source"
    for call in calls:
        assert call.rstrip().endswith(", 0, 1);"), f"Unexpected swzl trans tail: {call}"

    generator = torch.Generator(device="cuda").manual_seed(0)
    data = torch.randint(-128, 128, shape, generator=generator, device="cuda", dtype=torch.int32).to(torch.bfloat16)
    out = kernel(data)
    torch.testing.assert_close(out, data[:, :16], rtol=0, atol=0)


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_swzl_b16_half_bank_and_slices():
    """bf16, 64B swizzle (mode 1), multi-iteration K-stacked fragments,
    plus non-zero column-slice mins.

    Each x4 packet covers a 16x16 tile.
    """
    _run_raw_case("bfloat16", (32, 32), (16, 16), make_half_bank_swizzled_layout, 1, row_major=True)
    _run_raw_case("bfloat16", (32, 32), (16, 32), make_half_bank_swizzled_layout, 1, row_major=False)
    _run_raw_case("bfloat16", (16, 64), (16, 32), make_full_bank_swizzled_layout, 0, row_major=False)
    _run_raw_case("bfloat16", (16, 64), (16, 16), make_full_bank_swizzled_layout, 0, sliced=(0, 16))
    _run_raw_case("bfloat16", (16, 64), (16, 16), make_full_bank_swizzled_layout, 0, sliced=(0, 32))


@tilelang.testing.requires_ppu_compute_version(1, 5)
@pytest.mark.parametrize(
    "swizzle_fn,shared_shape",
    [(make_full_bank_swizzled_layout, (32, 64)), (make_half_bank_swizzled_layout, (32, 32))],
)
@pytest.mark.parametrize("frag_shape,row_major", [((8, 32), True), ((16, 32), True), ((32, 16), False)])
@pytest.mark.parametrize("transposed", [False, True])
def test_swzl_incompatible_fragment_fallback(swizzle_fn, shared_shape, frag_shape, row_major, transposed):
    """Four horizontal/vertical 8x8 blocks cannot use the 2x2 bulk helper.

    Both the shared allocation and the larger copy regions can have heights
    divisible by 16 while each instruction packet still has the wrong shape.
    """
    kernel = tilelang.compile(
        _make_raw_kernel("bfloat16", shared_shape, frag_shape, swizzle_fn, row_major=row_major, transposed=transposed),
        out_idx=[-1],
    )
    source = kernel.get_kernel_source()
    assert SWZL not in source, "Incompatible fragment packet unexpectedly used a SWZL bulk load"
    helper = "tl::tix_ldmatrix_x4_trans(" if transposed else "tl::tix_ldmatrix_x4("
    assert helper in source, "Expected the ordinary per-row matrix-load fallback"

    # Random data varies across both dimensions to expose incorrect source rectangles.
    generator = torch.Generator(device="cuda").manual_seed(0)
    data = torch.randint(-128, 128, shared_shape, generator=generator, device="cuda", dtype=torch.int32).to(torch.bfloat16)
    out = kernel(data)
    ref = data[: frag_shape[0], : frag_shape[1]]
    torch.testing.assert_close(out.view(torch.int16), ref.view(torch.int16), rtol=0, atol=0)


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_swzl_b8_full_bank():
    """fp8 e4m3 and int8 no-trans copies use the packed-b16 swizzle path."""
    for dtype in ["float8_e4m3fn", "int8"]:
        _run_b8_raw(dtype, make_full_bank_swizzled_layout, 0)


def _run_b8_raw(dtype, swizzle_fn, mode):
    shared_shape, frag_shape = (16, 128), (16, 32)
    frag_layout = _frag_8x16(frag_shape[0] // 8, frag_shape[1] // 16)

    @T.prim_func
    def main(A: T.Tensor(shared_shape, dtype), B: T.Tensor(frag_shape, dtype)):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared(shared_shape, dtype)
            fragment = T.alloc_fragment(frag_shape, dtype)
            T.annotate_layout({shared: swizzle_fn(shared), fragment: frag_layout})
            T.copy(A, shared)
            T.copy(shared[0 : frag_shape[0], 0 : frag_shape[1]], fragment)
            T.copy(fragment, B)

    kernel = tilelang.compile(main, out_idx=[-1])
    source = kernel.get_kernel_source()
    calls = re.findall(re.escape(SWZL) + r"\([^;]+;", source)
    assert len(calls) > 0, f"Expected {SWZL} in generated CUDA source"
    for call in calls:
        assert call.rstrip().endswith(f", {mode}, 1);"), f"Unexpected swzl tail: {call}"

    generator = torch.Generator(device="cuda").manual_seed(0)
    raw = torch.randint(0, 256, shared_shape, generator=generator, device="cuda", dtype=torch.uint8)
    out = kernel(raw.view(torch.float8_e4m3fn) if dtype == "float8_e4m3fn" else raw.to(torch.int8))
    torch.testing.assert_close(out.view(torch.uint8), raw[:, :32], rtol=0, atol=0)


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_swzl_fallback_locked():
    """Linear / quarter-bank (32B) layouts must NOT use the swzl instruction."""
    for swizzle_fn in [make_linear_layout, make_quarter_bank_swizzled_layout]:
        kernel = tilelang.compile(_make_raw_kernel("bfloat16", (16, 64), (16, 16), swizzle_fn), out_idx=[-1])
        source = kernel.get_kernel_source()
        assert SWZL not in source, f"swzl unexpectedly used for {swizzle_fn.__name__}"
        # The copy still lowers (plain ldmatrix or normal copy) and stays exact.
        g = torch.Generator(device="cuda").manual_seed(0)
        data = (torch.randn((16, 64), generator=g, device="cuda") * 4).bfloat16()
        out = kernel(data)
        torch.testing.assert_close(out, data[:, :16], rtol=0, atol=0)


def _make_gemm_rr(dtype, operand):
    """GEMM-RR with swizzled shared: TN exercises swzl on the non-transposed
    operand and (for b8) the m16n16.trans.b8 path on the transposed one."""
    accum_dtype = "float32"
    transposed = operand == "A"
    a_shape = (64, 128) if transposed else (128, 64)
    b_shape = (128, 64) if transposed else (64, 128)
    a_fragment_shape = (32, 128) if transposed else (128, 32)
    b_fragment_shape = (128, 32) if transposed else (32, 128)

    @T.prim_func
    def main(
        A: T.Tensor(a_shape, dtype),
        B: T.Tensor(b_shape, dtype),
        C: T.Tensor((128, 128), accum_dtype),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared(a_shape, dtype)
            B_shared = T.alloc_shared(b_shape, dtype)
            A_fragment = T.alloc_fragment(a_fragment_shape, dtype)
            B_fragment = T.alloc_fragment(b_fragment_shape, dtype)
            C_fragment = T.alloc_fragment((128, 128), accum_dtype)
            T.annotate_layout({A_shared: make_swizzled_layout(A_shared), B_shared: make_swizzled_layout(B_shared)})
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.clear(C_fragment)
            for ki in T.serial(2):
                if transposed:
                    T.copy(A_shared[ki * 32 : (ki + 1) * 32, :], A_fragment)
                    T.copy(B_shared[:, ki * 32 : (ki + 1) * 32], B_fragment)
                else:
                    T.copy(A_shared[:, ki * 32 : (ki + 1) * 32], A_fragment)
                    T.copy(B_shared[ki * 32 : (ki + 1) * 32, :], B_fragment)
                T.gemm(A_fragment, B_fragment, C_fragment, transpose_A=transposed, transpose_B=transposed)
            T.copy(C_fragment, C)

    return main


def _run_gemm_rr(dtype, operand):
    kernel = tilelang.compile(_make_gemm_rr(dtype, operand), out_idx=[2])
    source = kernel.get_kernel_source()
    calls = re.findall(re.escape(SWZL) + r"\([^;]+;", source)
    assert len(calls) > 0, f"Expected {SWZL} in generated CUDA source"
    if dtype != "bfloat16":
        assert SWZL_TRANS + "(" not in source, f"Unexpected trans swzl for {dtype}"

    td = {"float8_e4m3fn": torch.float8_e4m3fn, "bfloat16": torch.bfloat16}[dtype]
    g = torch.Generator().manual_seed(0)
    transposed = operand == "A"
    a_shape = (64, 128) if transposed else (128, 64)
    b_shape = (128, 64) if transposed else (64, 128)
    a = torch.randint(-4, 5, a_shape, generator=g).to(td)
    b = torch.randint(-4, 5, b_shape, generator=g).to(td)
    reference = a.float().T @ b.float().T if transposed else a.float() @ b.float()
    result = kernel(a.cuda(), b.cuda())
    torch.testing.assert_close(result.float().cpu(), reference, rtol=0, atol=0)


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_swzl_gemm_rr_fp8_and_bf16():
    """Both fragment directions stay correct; b8 no-trans still uses swizzle."""
    _run_gemm_rr("float8_e4m3fn", "B")
    _run_gemm_rr("float8_e4m3fn", "A")
    _run_gemm_rr("bfloat16", "B")
    _run_gemm_rr("bfloat16", "A")


if __name__ == "__main__":
    tilelang.testing.main()
