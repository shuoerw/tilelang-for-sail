"""Automatic AIU selection preserves synchronous and pipelined copy ordering."""

import pytest

import tilelang.ppu.language as T
from tilelang import tvm
from tilelang.engine.lower import lower
from tilelang.layout import make_full_bank_swizzled_layout


TARGET = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})


def _kernel(num_stages, prefer_instruction):
    @T.prim_func
    def main(A: T.Tensor((4, 16, 128), "float16"), B: T.Tensor((4, 16, 128), "float16")):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((16, 128), "float16")
            T.annotate_layout({shared: make_full_bank_swizzled_layout(shared)})
            for k in T.Pipelined(4, num_stages=num_stages):
                T.copy(A[k, :, :], shared, prefer_instruction=prefer_instruction)
                T.copy(shared, B[k, :, :])

    return main


@pytest.mark.parametrize("num_stages", [0, 2])
@pytest.mark.parametrize("enabled", [False, True])
def test_automatic_aiu_matches_explicit_preference(num_stages, enabled):
    with TARGET, tvm.transform.PassContext(opt_level=3, config={"tl.disable_aiu_lower": not enabled}):
        automatic = lower(_kernel(num_stages, None), target=TARGET).kernel_source
        preferred = lower(_kernel(num_stages, "aiu"), target=TARGET).kernel_source

    assert automatic == preferred
    assert ("tl::aiu_load(" in automatic) == enabled
    if enabled:
        assert automatic.count("tl::aiu_load(") == 2 * automatic.count("tl::cp_async_commit()")
        assert "tl::cp_async_wait" in automatic
        assert "__syncthreads()" in automatic


@pytest.mark.parametrize("num_stages", [0, 2])
def test_pipeline_cp_async_preference_prevents_automatic_aiu(num_stages):
    with TARGET, tvm.transform.PassContext(opt_level=3, config={"tl.disable_aiu_lower": False}):
        automatic = lower(_kernel(num_stages, None), target=TARGET).kernel_source
        source = lower(_kernel(num_stages, "cp_async"), target=TARGET).kernel_source

    assert "tl::aiu_load(" in automatic
    assert "tl::aiu_load(" not in source


@pytest.mark.parametrize("enabled", [False, True])
def test_plain_copy_interface_matches_automatic_selection(enabled):
    from testing.python.kernel.test_tilelang_kernel_aiu_copy_interface import _kernel as interface_kernel

    with TARGET, tvm.transform.PassContext(opt_level=3, config={"tl.disable_aiu_lower": not enabled}):
        source = lower(interface_kernel("plain"), target=TARGET).kernel_source
    assert ("tl::aiu_load(" in source) == enabled
