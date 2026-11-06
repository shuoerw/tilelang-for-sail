"""Check complete lowering of PPU copy edge cases without a device."""

import pytest

from tilelang import tvm
from tilelang.engine.lower import lower
from testing.python.ppu.transform.test_ppu_copy_correctness import (
    TARGET,
    make_fp4_scalar_copy_kernel,
    make_fp4_fallback_kernel,
    make_fp4_packed_copy_kernel,
    make_odd_width_kernel,
    make_region_kernel,
    make_thread_dependent_fp4_kernel,
)


def _source(program, **config):
    with TARGET, tvm.transform.PassContext(opt_level=3, config={"tl.disable_aiu_lower": False, **config}):
        return lower(program, target=TARGET).kernel_source


@pytest.mark.parametrize("width", [255, 257])
def test_odd_fp4_width_compiles(width):
    source = _source(make_odd_width_kernel(width))
    assert "tl::aiu_load(" not in source
    assert "tl_fp4_packed_store(" in source


@pytest.mark.parametrize("dtype", ["float16", "float4_e2m1fn"])
def test_aiu_copy_with_nonzero_thread_and_region_origins_compiles(dtype):
    source = _source(make_region_kernel(dtype, 128, 8, 1, 128, 128, leading=True))
    assert "tl::aiu_load(" in source


@pytest.mark.parametrize("stride,offset,single_thread", [(1, 1, False), (2, 0, False), (2, 1, False), (1, 1, True)])
def test_fp4_scalar_copy_compiles(stride, offset, single_thread):
    source = _source(make_fp4_scalar_copy_kernel(stride, offset, single_thread))
    assert "tl_fp4_packed_store(" in source


def test_fp4_fallback_preserves_cp_async():
    source = _source(make_fp4_packed_copy_kernel())
    assert "tl::cp_async" in source
    assert "tl::aiu_load(" not in source
    assert "tl_fp4_packed_store(" not in source


@pytest.mark.parametrize("dependent", ["source", "destination"])
@pytest.mark.parametrize("thread_start", [0, 128])
def test_thread_dependent_fp4_copy_has_scoped_synchronization(dependent, thread_start):
    source = _source(make_thread_dependent_fp4_kernel(dependent, thread_start))
    assert "tl_fp4_packed_store(" in source
    if thread_start:
        assert "tl::__sync_thread_partial" in source
        assert "__syncthreads()" not in source
    else:
        assert "__syncthreads()" in source


@pytest.mark.parametrize("thread_start", [0, 128])
@pytest.mark.parametrize("dtype", ["float16", "float4_e2m1fn"])
def test_global_fp4_copy_synchronizes_participating_threads(thread_start, dtype):
    source = _source(make_fp4_fallback_kernel(dtype, "global", thread_start))
    assert "tl_fp4_packed_store(" in source
    if thread_start:
        assert source.count("tl::__sync_thread_partial") >= 2
        assert "__syncthreads()" not in source
    else:
        assert source.count("__syncthreads()") >= 2


def test_fp4_scalar_copy_requires_sync_pass():
    with pytest.raises(tvm.error.InternalError, match="Packed FP4 scalar copy fallback requires ThreadSync"):
        _source(make_fp4_fallback_kernel("float16", "global"), **{"tl.disable_thread_storage_sync": True})
