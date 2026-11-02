"""PPU 1.5 numerical regressions for AIU destination regions."""

import pytest
import torch

import tilelang
import tilelang.testing
from tilelang import tvm
from testing.python.ppu.transform.test_ppu_copy_correctness import (
    make_region_kernel,
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
