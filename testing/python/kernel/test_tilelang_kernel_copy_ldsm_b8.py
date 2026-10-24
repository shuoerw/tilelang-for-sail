"""Byte-exact PPU b8 copies: x2/x4 matrix loads and the x1 fallback.

The 8x16 fragment maps four bytes to each thread. Larger fragments can use
b16 x2/x4 loads; a single matrix uses ordinary copying. Transposed fragments
and sliced, pipelined shared buffers are covered separately.
"""

import re

import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import make_gemm_fragment_8x8_transposed, make_linear_layout

DTYPES = ["int8", "float8_e4m3fn", "float8_e5m2"]
TORCH_DTYPES = {
    "int8": torch.int8,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}


def _fragment_8x16(rep_rows, rep_cols):
    """Fragment equivalent to MakeGemmFragment8x16PPU: thread=j//4+4*i, index=j%4."""
    base = T.Fragment((8, 16), forward_fn=lambda i, j: (j // 4 + 4 * i, j % 4))
    return base.repeat([rep_rows, rep_cols], repeat_on_thread=False)


def _make_b8_copy_kernel(dtype, rep_rows, rep_cols):
    """Raw b8 S->R copy into an explicitly annotated 8x16 fragment layout."""
    rows, cols = 8 * rep_rows, 16 * rep_cols
    fragment_layout = _fragment_8x16(rep_rows, rep_cols)

    @T.prim_func
    def main(A: T.Tensor((rows, cols), dtype), B: T.Tensor((rows, cols), dtype)):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((rows, cols), dtype)
            fragment = T.alloc_fragment((rows, cols), dtype)
            T.annotate_layout({shared: make_linear_layout(shared), fragment: fragment_layout})
            T.copy(A, shared)
            T.copy(shared, fragment)
            T.copy(fragment, B)

    return main


def _rand_bytes(shape):
    generator = torch.Generator(device="cuda").manual_seed(0)
    return torch.randint(0, 256, shape, generator=generator, device="cuda", dtype=torch.uint8)


def _check_b8_copy(dtype, rep_rows, rep_cols, expect_num):
    kernel = tilelang.compile(_make_b8_copy_kernel(dtype, rep_rows, rep_cols), out_idx=[-1])
    source = kernel.get_kernel_source()

    if expect_num is None:
        assert "tix_ldmatrix" not in source
    else:
        helper = f"tl::tix_ldmatrix_x{expect_num}"
        assert helper + "(" in source, f"Expected {helper} in generated CUDA source"
    assert "tix_ldmatrix_swzl" not in source, "Linear shared layout should not use swzl ldmatrix"
    assert not re.findall(r"tl::tix_ldmatrix\w*_trans\(", source), "b8 must use the non-trans ldmatrix"

    rows, cols = 8 * rep_rows, 16 * rep_cols
    raw = _rand_bytes((rows, cols))
    out = kernel(raw.view(TORCH_DTYPES[dtype]))
    # Compare raw bytes; keeps FP8 NaN encodings exact.
    torch.testing.assert_close(out.view(torch.uint8), raw, rtol=0, atol=0)


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_ldsm_b8_nontrans_x2_x4():
    """b8 non-trans ldmatrix: x2 (extent%8==0) and x4 (extent%16==0) shapes."""
    for dtype in DTYPES:
        _check_b8_copy(dtype, rep_rows=2, rep_cols=1, expect_num=2)
        _check_b8_copy(dtype, rep_rows=2, rep_cols=2, expect_num=4)


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_ldsm_b8_single_matrix_fallback():
    """A single 8x16 b8 matrix must copy every bit without the x1 template."""
    for dtype in DTYPES:
        _check_b8_copy(dtype, rep_rows=1, rep_cols=1, expect_num=None)


def _make_pipelined_b8_sliced_kernel(dtype):
    """Pipelined b8 S->R copy: multi-buffered shared (dynamic stage dim) plus
    a slice with non-zero row/col mins, into an 8x16-fragment (x4 shape)."""
    k_iters = 4
    fragment_layout = _fragment_8x16(2, 2)  # fragment (16, 32), extent 16 -> x4

    @T.prim_func
    def main(
        A: T.Tensor((k_iters, 24, 48), dtype),
        B: T.Tensor((k_iters, 16, 32), dtype),
    ):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((24, 48), dtype)
            fragment = T.alloc_fragment((16, 32), dtype)
            T.annotate_layout({shared: make_linear_layout(shared), fragment: fragment_layout})
            for ko in T.Pipelined(k_iters, num_stages=2):
                T.copy(A[ko, :, :], shared)
                T.copy(shared[4:20, 16:48], fragment)
                T.copy(fragment, B[ko, :, :])

    return main


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_ldsm_b8_pipelined_sliced_multibuffer():
    dtype = "int8"
    kernel = tilelang.compile(_make_pipelined_b8_sliced_kernel(dtype), out_idx=[-1])
    source = kernel.get_kernel_source()

    # The shared address must stitch the dynamic stage dim and the slice mins
    # ((4, 16) here) into the ldmatrix pointer.
    calls = re.findall(r"tl::tix_ldmatrix_x4\([^;]+\)", source)
    assert len(calls) > 0, "Expected tl::tix_ldmatrix_x4 in generated CUDA source; pipelined sliced b8 copy fell back to normal copy"

    raw = _rand_bytes((4, 24, 48))
    out = kernel(raw.view(TORCH_DTYPES[dtype]))
    ref = raw[:, 4:20, 16:48]
    torch.testing.assert_close(out.view(torch.uint8), ref, rtol=0, atol=0)


def _make_b8_transposed_copy_kernel(dtype):
    """b8 copy with a transposed 8x8 (b16-style) fragment: must NOT match the
    non-trans 8x16 matcher, so it stays on the normal copy path (transposed b8
    ldmatrix support is intentionally out of scope here)."""
    fragment_layout = make_gemm_fragment_8x8_transposed().repeat([2, 1], repeat_on_thread=False)

    @T.prim_func
    def main(A: T.Tensor((16, 8), dtype), B: T.Tensor((16, 8), dtype)):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((16, 8), dtype)
            fragment = T.alloc_fragment((16, 8), dtype)
            T.annotate_layout({shared: make_linear_layout(shared), fragment: fragment_layout})
            T.copy(A, shared)
            T.copy(shared, fragment)
            T.copy(fragment, B)

    return main


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_ldsm_b8_transposed_still_fallback():
    for dtype in DTYPES:
        kernel = tilelang.compile(_make_b8_transposed_copy_kernel(dtype), out_idx=[-1])
        source = kernel.get_kernel_source()
        assert "tix_ldmatrix" not in source, f"b8 transposed copy unexpectedly used ldmatrix for {dtype}"
        raw = _rand_bytes((16, 8))
        out = kernel(raw.view(TORCH_DTYPES[dtype]))
        torch.testing.assert_close(out.view(torch.uint8), raw, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
