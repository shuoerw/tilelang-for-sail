"""Numerical coverage for AIU layout selection and fallback copies."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from testing.python.ppu.transform.test_ppu_aiu_layout import _layout


@tilelang.testing.requires_ppu_compute_version(1, 5)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float8_e4m3fn", "float8_e5m2", "float4_e2m1fn"])
@pytest.mark.parametrize("bank_bytes", [64, 128])
@pytest.mark.parametrize("threads", [32, 128])
@pytest.mark.parametrize("kind", ["canonical", "flat", "row_major", "offset"])
def test_aiu_copy_preserves_values(dtype, bank_bytes, threads, kind):
    bits = tvm.DataType(dtype).bits
    height, width = 16, 2 * bank_bytes * 8 // bits

    @T.prim_func
    def main(A: T.Tensor((height, 2 * width), dtype), B: T.Tensor((height, width), dtype)):
        with T.Kernel(1, threads=threads):
            shared = T.alloc_shared((height, width), dtype)
            T.annotate_layout({shared: _layout(shared, bank_bytes, kind)})
            T.copy(A[:, width : 2 * width], shared, prefer_instruction="aiu")
            T.copy(shared, B)

    kernel = tilelang.compile(main, out_idx=[1], pass_configs={"tl.disable_aiu_lower": False})
    source = kernel.get_kernel_source()
    assert ("tl::aiu_load(" in source) == (kind in ("canonical", "flat"))
    byte_width = 2 * width * bits // 8
    generator = torch.Generator(device="cuda").manual_seed(7)
    raw = torch.randint(0, 256, (height, byte_width), device="cuda", dtype=torch.uint8, generator=generator)
    storage_dtype = torch.int8 if bits == 4 else getattr(torch, dtype)
    values = raw.view(storage_dtype)
    result = kernel(values).view(torch.uint8)
    expected = raw[:, byte_width // 2 :]
    torch.testing.assert_close(result, expected, rtol=0, atol=0)


@tilelang.testing.requires_ppu_compute_version(1, 5)
@pytest.mark.parametrize("threads", [32, 128])
def test_fp4_odd_offset_fallback_preserves_bytes(threads):
    test_aiu_copy_preserves_values("float4_e2m1fn", 128, threads, "odd_offset")


@tilelang.testing.requires_ppu_compute_version(1, 5)
@pytest.mark.parametrize("kind", ["row_major", "odd_offset"])
def test_fp4_pipeline_fallback_preserves_bytes(kind):
    dtype = "float4_e2m1fn"
    height, width, iterations = 16, 512, 3

    @T.prim_func
    def main(A: T.Tensor((iterations, height, 2 * width), dtype), B: T.Tensor((iterations, height, width), dtype)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((height, width), dtype)
            T.annotate_layout({shared: _layout(shared, 128, kind)})
            for k in T.Pipelined(iterations, num_stages=2):
                T.copy(A[k, :, width : 2 * width], shared, prefer_instruction="aiu")
                T.copy(shared, B[k, :, :])

    kernel = tilelang.compile(main, out_idx=[1], pass_configs={"tl.disable_aiu_lower": False})
    assert "tl::aiu_load(" not in kernel.get_kernel_source()
    generator = torch.Generator(device="cuda").manual_seed(13)
    raw = torch.randint(0, 256, (iterations, height, width), dtype=torch.uint8, device="cuda", generator=generator)
    result = kernel(raw.view(torch.int8)).view(torch.uint8)
    torch.testing.assert_close(result, raw[:, :, width // 2 :], rtol=0, atol=0)
