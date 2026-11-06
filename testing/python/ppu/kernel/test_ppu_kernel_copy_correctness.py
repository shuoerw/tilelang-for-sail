"""PPU 1.5 numerical regressions for AIU destination regions."""

import pytest
import torch

import tilelang
import tilelang.testing
import tilelang.ppu.language as T
from tilelang import tvm
from testing.python.ppu.transform.test_ppu_copy_correctness import (
    make_region_kernel,
    make_fp4_fallback_kernel,
    make_odd_width_kernel,
    make_thread_dependent_fp4_kernel,
)


@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
@pytest.mark.parametrize("dtype", ["float16", "float8_e4m3fn", "float4_e2m1fn"])
@pytest.mark.parametrize("bank_bytes", [64, 128])
@pytest.mark.parametrize("row_start,col_blocks,thread_start,threads,leading", [(0, 0, 0, 32, False), (8, 1, 128, 128, True)])
def test_aiu_subregion_preserves_bytes(dtype, bank_bytes, row_start, col_blocks, thread_start, threads, leading):
    program = make_region_kernel(dtype, bank_bytes, row_start, col_blocks, thread_start, threads, leading)
    kernel = tilelang.compile(program, out_idx=[1], pass_configs={"tl.disable_aiu_lower": False})
    assert "tl::aiu_load(" in kernel.get_kernel_source()
    raw = torch.randint(0, 256, (8, 2 * bank_bytes), device="cuda", dtype=torch.uint8)
    storage_dtype = torch.int8 if tvm.DataType(dtype).bits == 4 else getattr(torch, dtype)
    result = kernel(raw.view(storage_dtype)).view(torch.uint8)
    torch.testing.assert_close(result, raw, rtol=0, atol=0)


FP4_VALUES = [0, 0.5, 1, 1.5, 2, 3, 4, 6, 0, -0.5, -1, -1.5, -2, -3, -4, -6]


def _unpack(raw):
    return torch.stack((raw & 15, raw >> 4), dim=-1).flatten(-2)


@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
@pytest.mark.parametrize("dtype", ["float16", "float4_e2m1fn"])
@pytest.mark.parametrize("destination", ["shared", "global"])
@pytest.mark.parametrize("thread_start", [0, 128])
def test_scalar_fp4_fallback_preserves_values(dtype, destination, thread_start):
    kernel = tilelang.compile(
        make_fp4_fallback_kernel(dtype, destination, thread_start), out_idx=[2], pass_configs={"tl.disable_aiu_lower": False}
    )
    codes = torch.arange(64, device="cuda", dtype=torch.uint8) % 16
    expected = torch.tensor(FP4_VALUES, device="cuda", dtype=torch.float16)[codes.long()]
    source = expected if dtype == "float16" else (codes[0::2] | (codes[1::2] << 4)).view(torch.int8)
    packed = torch.full((33,), 0x55, device="cuda", dtype=torch.uint8)
    torch.testing.assert_close(kernel(source, packed.view(torch.int8)), expected, rtol=0, atol=0)
    if destination == "global":
        result_codes = _unpack(packed)
        assert int(result_codes[0]) == 5
        assert int(result_codes[-1]) == 5


@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
@pytest.mark.parametrize("width", [255, 257])
def test_odd_fp4_shared_copy_preserves_values(width):
    kernel = tilelang.compile(make_odd_width_kernel(width), out_idx=[1], pass_configs={"tl.disable_aiu_lower": False})
    raw = torch.randint(0, 256, (16, 256), device="cuda", dtype=torch.uint8)
    expected = torch.tensor(FP4_VALUES, device="cuda", dtype=torch.float16)[_unpack(raw).long()][:, :width]
    torch.testing.assert_close(kernel(raw.view(torch.int8)), expected, rtol=0, atol=0)


@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
@pytest.mark.parametrize("dependent", ["source", "destination"])
@pytest.mark.parametrize("thread_start", [0, 128])
def test_thread_dependent_fp4_copy_preserves_values(dependent, thread_start):
    kernel = tilelang.compile(
        make_thread_dependent_fp4_kernel(dependent, thread_start), out_idx=[1], pass_configs={"tl.disable_aiu_lower": False}
    )
    codes = torch.arange(96, device="cuda", dtype=torch.uint8) % 16
    source = (codes[0::2] | (codes[1::2] << 4)).view(torch.int8)
    decoded = torch.tensor(FP4_VALUES, device="cuda", dtype=torch.float16)[codes.long()]
    expected = decoded[:96:3] if dependent == "source" else decoded[:32]
    torch.testing.assert_close(kernel(source), expected, rtol=0, atol=0)


@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
@pytest.mark.parametrize(
    "width,prefix",
    [
        (64, 0),
        pytest.param(
            65,
            0,
            marks=pytest.mark.xfail(
                reason="Different CTAs can update separate FP4 nibbles of the same global byte", raises=AssertionError, strict=False
            ),
        ),
        pytest.param(
            64,
            1,
            marks=pytest.mark.xfail(
                reason="Different CTAs can update separate FP4 nibbles of the same global byte", raises=AssertionError, strict=False
            ),
        ),
    ],
)
def test_multicta_fp4_global_copy_preserves_values(width, prefix):
    blocks = 64
    elements = (blocks * width + prefix + 1) // 2 * 2

    @T.prim_func
    def main(A: T.Tensor((elements,), "float4_e2m1fn"), B: T.Tensor((elements,), "float4_e2m1fn")):
        with T.Kernel(blocks, threads=32) as bx:
            start = bx * width + prefix
            T.copy(A[start : start + width], B[start : start + width])

    kernel = tilelang.compile(main, out_idx=[1])
    raw = torch.randint(0, 256, (elements // 2,), device="cuda", dtype=torch.uint8)
    expected = _unpack(raw)[prefix : prefix + blocks * width]
    for _ in range(20):
        actual = _unpack(kernel(raw.view(torch.int8)).view(torch.uint8))[prefix : prefix + blocks * width]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
