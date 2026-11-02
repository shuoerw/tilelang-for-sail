"""Check complete lowering of PPU copy edge cases without a device."""

import pytest

from tilelang import tvm
from tilelang.engine.lower import lower
from testing.python.ppu.transform.test_ppu_copy_correctness import (
    TARGET,
    make_odd_width_kernel,
    make_region_kernel,
)


def _source(program, **config):
    with TARGET, tvm.transform.PassContext(opt_level=3, config={"tl.disable_aiu_lower": False, **config}):
        return lower(program, target=TARGET).kernel_source


@pytest.mark.parametrize("width", [255, 257])
def test_odd_fp4_width_reports_unsupported_fallback(width):
    with pytest.raises(tvm.error.InternalError, match="AIU FP4 fallback requires all stores to use cp.async"):
        _source(make_odd_width_kernel(width))


@pytest.mark.parametrize("dtype", ["float16", "float4_e2m1fn"])
def test_aiu_copy_with_nonzero_thread_and_region_origins_compiles(dtype):
    source = _source(make_region_kernel(dtype, 128, 8, 1, 128, 128, leading=True))
    assert "tl::aiu_load(" in source
