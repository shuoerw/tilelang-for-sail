"""Test LowerLDSM addressing for sliced / multi-buffered shared buffers on PPU.

A b16 shared->register copy is emitted as ``tl::tix_ldmatrix[_trans]`` only
when the shared address is stitched correctly: leading stage dims come from
the copy range and slice offsets (range.min) are added to the spatial
coordinates. When the pipeline transform multi-buffers a shared tensor, the
copy range keeps a dynamic stage index plus non-zero slice mins; before this
fix such copies were silently lowered through the normal copy path (no
ldmatrix) and the fragment-inverse coordinates were used as-is. These tests
pin the ldmatrix emission and the numerics.

Coverage: b16 (float16) S->R copies via ``T.copy`` (raw, statically sliced,
and software-pipelined stage-buffered), plus a pipelined fp16 GEMM regression
under both the default swizzled LDSM path and the explicitly disabled path.
32 threads are used for the raw cases and 128 for the GEMM case.
"""

import re

import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.transform import PassConfigKey
from tilelang.layout import (
    make_gemm_fragment_8x8,
    make_gemm_fragment_8x8_transposed,
    make_linear_layout,
)

DTYPE = "float16"


def _make_pipelined_sliced_copy_kernel():
    """S->R copy out of a software-pipelined shared buffer.

    num_stages=2 multi-buffers ``shared`` into (2, 16, 32) with a dynamic
    stage dim, and the S->R copy reads the slice rows [4:12] x cols [16:32]
    (both range mins > 0) into an 8x16 mma fragment.
    """
    k_iters = 4
    shared_rows, shared_cols = 16, 32
    row_start, col_start = 4, 16
    rows, cols = 8, 16
    fragment_layout = make_gemm_fragment_8x8().repeat([rows // 8, cols // 8], repeat_on_thread=False)

    @T.prim_func
    def main(
        A: T.Tensor((k_iters, shared_rows, shared_cols), DTYPE),
        B: T.Tensor((k_iters, rows, cols), DTYPE),
    ):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((shared_rows, shared_cols), DTYPE)
            fragment = T.alloc_fragment((rows, cols), DTYPE)
            T.annotate_layout({shared: make_linear_layout(shared), fragment: fragment_layout})
            for ko in T.Pipelined(k_iters, num_stages=2):
                T.copy(A[ko, :, :], shared)
                T.copy(shared[row_start : row_start + rows, col_start : col_start + cols], fragment)
                T.copy(fragment, B[ko, :, :])

    return main


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_ldsm_b16_pipelined_sliced_multibuffer():
    kernel = tilelang.compile(_make_pipelined_sliced_copy_kernel(), out_idx=[-1])
    source = kernel.get_kernel_source()

    # Before the fix this copy produced no ldmatrix at all (LowerNormal
    # fallback); the fixed lowering must stitch the stage dim and the slice
    # offsets into the shared address.
    calls = re.findall(r"tl::tix_ldmatrix_x\d+\(", source)
    assert len(calls) > 0, "Expected tl::tix_ldmatrix in generated CUDA source; pipelined sliced S->R copy fell back to normal copy"

    k_iters, row_start, col_start, rows, cols = 4, 4, 16, 8, 16
    data = torch.randn(k_iters, 16, 32, device="cuda").half() * 4
    out = kernel(data)
    ref = data[:, row_start : row_start + rows, col_start : col_start + cols]
    # A copy is exact; compare raw bits.
    torch.testing.assert_close(out.view(torch.int16), ref.view(torch.int16), rtol=0, atol=0)


def _make_static_copy_kernel(sliced, transposed):
    """Raw S->R copy into an explicitly annotated mma 8x8 fragment layout.

    sliced=True uses a (2, 48, 32) shared buffer with a leading stage dim and
    a copy region starting at (stage=1, row=4, col=16) > 0; the mins are
    compile-time constants.
    """
    rows, cols = 16, 16
    stage, row_start, col_start = 1, 4, 16
    shared_shape = (2, 48, 32) if sliced else (rows, cols)
    base_layout = make_gemm_fragment_8x8_transposed() if transposed else make_gemm_fragment_8x8()
    fragment_layout = base_layout.repeat([rows // 8, cols // 8], repeat_on_thread=False)

    @T.prim_func
    def main(
        A: T.Tensor(shared_shape, DTYPE),
        B: T.Tensor((rows, cols), DTYPE),
    ):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared(shared_shape, DTYPE)
            fragment = T.alloc_fragment((rows, cols), DTYPE)
            T.annotate_layout({shared: make_linear_layout(shared), fragment: fragment_layout})
            T.copy(A, shared)
            if sliced:
                region = shared[stage, row_start : row_start + rows, col_start : col_start + cols]
            else:
                region = shared[0:rows, 0:cols]
            T.copy(region, fragment)
            T.copy(fragment, B)

    return main


def _run_static_copy_case(sliced, transposed):
    kernel = tilelang.compile(_make_static_copy_kernel(sliced, transposed), out_idx=[-1])
    source = kernel.get_kernel_source()

    helper = f"tl::tix_ldmatrix_x4{'_trans' if transposed else ''}"
    assert helper + "(" in source, f"Expected {helper} in generated CUDA source"
    assert "tix_ldmatrix_swzl" not in source, "Linear shared layout should not use swzl ldmatrix"

    rows, cols = 16, 16
    shape = (2, 48, 32) if sliced else (rows, cols)
    data = torch.randn(shape, device="cuda").half() * 4
    out = kernel(data)
    if sliced:
        ref = data[1, 4 : 4 + rows, 16 : 16 + cols]
    else:
        ref = data
    torch.testing.assert_close(out.view(torch.int16), ref.view(torch.int16), rtol=0, atol=0)


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_ldsm_b16_static_sliced_view():
    """Static sliced / stage-buffered shared view: trans and non-trans."""
    _run_static_copy_case(sliced=True, transposed=False)
    _run_static_copy_case(sliced=True, transposed=True)


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_ldsm_b16_nonsliced_still_ldmatrix():
    """Base case (no leading dims, zero range mins) keeps the ldmatrix path."""
    _run_static_copy_case(sliced=False, transposed=False)
    _run_static_copy_case(sliced=False, transposed=True)


def _make_pipelined_gemm_kernel():
    """Pipelined fp16 GEMM: the standard mma-macro ldmatrix path must keep
    working with multi-buffered (staged) shared operands."""
    M = N = K = 128
    block_M = block_N = 64
    block_K = 32

    @T.prim_func
    def main(
        A: T.Tensor((M, K), DTYPE),
        B: T.Tensor((N, K), DTYPE),
        C: T.Tensor((M, N), DTYPE),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), DTYPE)
            B_shared = T.alloc_shared((block_N, block_K), DTYPE)
            C_local = T.alloc_fragment((block_M, block_N), T.float32)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[bx * block_N, k * block_K], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_ldsm_b16_pipelined_gemm():
    kernel = tilelang.compile(_make_pipelined_gemm_kernel(), out_idx=[-1])
    source = kernel.get_kernel_source()

    # PPU15 enables the swizzled bulk-tensor LDSM by default.  Verify both
    # that default path and the ordinary ldmatrix path selected when callers
    # explicitly disable the optimization.
    assert "tl::tix_ldmatrix_swzl_x4(" in source, (
        "Expected default swizzled ldmatrix in generated CUDA source"
    )

    a = torch.randn(128, 128, device="cuda").half()
    b = torch.randn(128, 128, device="cuda").half()
    c = kernel(a, b)
    ref_c = a.to(torch.float32) @ b.to(torch.float32).t()
    torch.testing.assert_close(c.to(torch.float32), ref_c, rtol=2e-2, atol=2e-2)

    normal_kernel = tilelang.compile(
        _make_pipelined_gemm_kernel(),
        out_idx=[-1],
        pass_configs={PassConfigKey.TL_DISABLE_LDMAT_SWZL: True},
    )
    normal_source = normal_kernel.get_kernel_source()
    calls = re.findall(r"tl::tix_ldmatrix_x4\(", normal_source)
    assert len(calls) > 0, (
        "Expected ordinary tl::tix_ldmatrix_x4 when swizzle is disabled"
    )

    normal_c = normal_kernel(a, b)
    torch.testing.assert_close(
        normal_c.to(torch.float32), ref_c, rtol=2e-2, atol=2e-2
    )


if __name__ == "__main__":
    tilelang.testing.main()
