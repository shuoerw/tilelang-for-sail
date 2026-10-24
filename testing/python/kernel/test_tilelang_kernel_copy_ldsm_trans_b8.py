"""Test PPU0015 m16n16.trans.b8 LowerLDSM loads for T.copy (matrix A/B).

b8 transposed shared->register copies whose fragment matches the m16k32
A-operand or B-operand layout are emitted as
``tl::tix_ldmatrix_x2_trans_m16n16_b8`` (ldmatrix.x2.trans.m16n16.shared.b8).
The trailing ``trans_block`` argument selects the register order: ``false``
for the raw A order, ``true`` for B (registers 1 and 2 of each x2 packet
swapped). Before this change every one of these copies fell back to the
normal copy path (no ldmatrix).

Only the x2 form is emitted: the x1 m16n16.b8 form compiles but its register
distribution does not match the mma-operand order on this hardware (device
probe, see commit message), so num is fixed at 2 (one m16k32 packet per
iteration).

Coverage: int8, float8_e4m3fn, float8_e5m2 at raw-copy level (A-order
trans_block=false and B-order trans_block=true, statically sliced variants,
partial-fragment fallback); fp8 GEMM-RR (NN/TT) end to end.
"""

import re

import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import make_linear_layout, make_swizzled_layout

DTYPES = ["int8", "float8_e4m3fn", "float8_e5m2"]
TORCH_DTYPES = {
    "int8": torch.int8,
    "float8_e4m3fn": torch.float8_e4m3fn,
    "float8_e5m2": torch.float8_e5m2,
}
HELPER = "tl::tix_ldmatrix_x2_trans_m16n16_b8"


# fragment order equations (see A repo testing/python/cuda/
# test_cuda_ldmatrix_trans_b8.py): B exchanges registers 1 and 2.
def _a_order(rows):
    return T.Fragment(
        (rows, 16),
        forward_fn=lambda i, j: (4 * (j % 8) + (i % 16) // 4, 8 * (i // 16) + 4 * (j // 8) + i % 4),
    )


def _b_order(rows):
    return T.Fragment(
        (rows, 16),
        forward_fn=lambda i, j: (4 * (j % 8) + (i % 16) // 4, 4 * (i // 16) + 8 * (j // 8) + i % 4),
    )


def _rand_bytes(shape):
    generator = torch.Generator(device="cuda").manual_seed(0)
    return torch.randint(0, 256, shape, generator=generator, device="cuda", dtype=torch.uint8)


def _make_trans_copy_kernel(dtype, operand, sliced=False):
    """Raw b8 transposed S->R copy: fragment (32, 16) -> one x2 packet.

    sliced=True copies out of a (2, 48, 32) stage-buffered shared tensor at
    (stage=1, row=4, col=16), exercising the C1 range stitching jointly with
    the 16-byte row-alignment proof.
    """
    rows = 32
    fragment_layout = (_a_order if operand == "A" else _b_order)(rows)
    shared_shape = (2, 48, 32) if sliced else (rows, 16)
    stage, row_start, col_start = (1, 4, 16) if sliced else (0, 0, 0)

    @T.prim_func
    def main(A: T.Tensor(shared_shape, dtype), B: T.Tensor((rows, 16), dtype)):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared(shared_shape, dtype)
            fragment = T.alloc_fragment((rows, 16), dtype)
            T.annotate_layout({shared: make_linear_layout(shared), fragment: fragment_layout})
            T.copy(A, shared)
            if sliced:
                region = shared[stage, row_start : row_start + rows, col_start : col_start + 16]
            else:
                region = shared[0:rows, 0:16]
            T.copy(region, fragment)
            T.copy(fragment, B)

    return main


def _run_trans_copy_case(dtype, operand, sliced):
    kernel = tilelang.compile(_make_trans_copy_kernel(dtype, operand, sliced), out_idx=[-1])
    source = kernel.get_kernel_source()

    assert HELPER + "(" in source, f"Expected {HELPER} in generated CUDA source"
    flag = "false" if operand == "A" else "true"
    calls = re.findall(re.escape(HELPER) + r"\([^;]+\)", source)
    assert len(calls) > 0
    for call in calls:
        assert call.rstrip().endswith(", " + flag + ")"), f"Expected trans_block={flag} for operand {operand}, got: {call}"

    shape = (2, 48, 32) if sliced else (32, 16)
    raw = _rand_bytes(shape)
    out = kernel(raw.view(TORCH_DTYPES[dtype]))
    if sliced:
        ref = raw[1, 4:36, 16:32]
    else:
        ref = raw
    # Compare raw bytes; keeps FP8 NaN encodings exact.
    torch.testing.assert_close(out.view(torch.uint8), ref, rtol=0, atol=0)


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_ldsm_trans_b8_copy_operands():
    """A-order (trans_block=false) and B-order (trans_block=true), all dtypes."""
    for dtype in DTYPES:
        for operand in ["A", "B"]:
            _run_trans_copy_case(dtype, operand, sliced=False)


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_ldsm_trans_b8_sliced_multibuffer():
    for dtype in DTYPES:
        for operand in ["A", "B"]:
            _run_trans_copy_case(dtype, operand, sliced=True)


def _make_trans_b8_partial_kernel(dtype):
    """Fragment (16, 16): extent 8 % 16 != 0, must NOT emit m16n16 (num=2 only)."""
    fragment_layout = _a_order(16)

    @T.prim_func
    def main(A: T.Tensor((16, 16), dtype), B: T.Tensor((16, 16), dtype)):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((16, 16), dtype)
            fragment = T.alloc_fragment((16, 16), dtype)
            T.annotate_layout({shared: make_linear_layout(shared), fragment: fragment_layout})
            T.copy(A, shared)
            T.copy(shared, fragment)
            T.copy(fragment, B)

    return main


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_ldsm_trans_b8_partial_fragment_fallback():
    for dtype in DTYPES:
        kernel = tilelang.compile(_make_trans_b8_partial_kernel(dtype), out_idx=[-1])
        source = kernel.get_kernel_source()
        assert "m16n16_b8" not in source, f"Partial b8 fragment unexpectedly used m16n16 ldmatrix for {dtype}"
        raw = _rand_bytes((16, 16))
        out = kernel(raw.view(TORCH_DTYPES[dtype]))
        torch.testing.assert_close(out.view(torch.uint8), raw, rtol=0, atol=0)


def _make_gemm_rr_kernel(dtype, operand):
    """GEMM-RR fp8: NN exercises the B operand (trans_block=true order),
    TT exercises the A operand (trans_block=false order)."""
    accum_dtype = "int32" if dtype == "int8" else "float32"
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


def _check_gemm_rr(dtype, operand):
    kernel = tilelang.compile(_make_gemm_rr_kernel(dtype, operand), out_idx=[2])
    source = kernel.get_kernel_source()

    assert HELPER + "(" in source, f"Expected {HELPER} in generated CUDA source"
    flag = "false" if operand == "A" else "true"
    calls = re.findall(re.escape(HELPER) + r"\([^;]+\)", source)
    assert any(call.rstrip().endswith(", " + flag + ")") for call in calls), (
        f"Expected trans_block={flag} order in operand-{operand} GEMM-RR"
    )

    torch_dtype = TORCH_DTYPES[dtype]
    generator = torch.Generator().manual_seed(0)
    transposed = operand == "A"
    a_shape = (64, 128) if transposed else (128, 64)
    b_shape = (128, 64) if transposed else (64, 128)
    a = torch.randint(-4, 5, a_shape, generator=generator).to(torch_dtype)
    b = torch.randint(-4, 5, b_shape, generator=generator).to(torch_dtype)
    # Small integer values isolate layout correctness from MMA truncation.
    reference = a.float().T @ b.float().T if transposed else a.float() @ b.float()
    result = kernel(a.cuda(), b.cuda())
    torch.testing.assert_close(result.float().cpu(), reference, rtol=0, atol=0)


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_ldsm_trans_b8_gemm_rr():
    """NN (B-operand transposed) and TT (A-operand transposed) fp32-acc fp8."""
    _check_gemm_rr("float8_e4m3fn", "B")
    _check_gemm_rr("float8_e4m3fn", "A")


if __name__ == "__main__":
    tilelang.testing.main()
