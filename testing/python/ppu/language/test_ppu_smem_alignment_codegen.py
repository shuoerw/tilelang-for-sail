"""Explicit alignment for dynamic shared and unmerged static shared."""

import re

import pytest

import tilelang
import tilelang.language as T
from tilelang import tvm

pytestmark = pytest.mark.skipif(
    tvm.get_global_func("target.build.tilelang_ppu", allow_missing=True) is None,
    reason="Requires a build with PPU codegen",
)


@pytest.mark.parametrize("scope", ["shared", "shared.dyn"])
@pytest.mark.parametrize("alignment", [64, 128, 256])
@pytest.mark.parametrize("disable_reuse", [False, True])
def test_explicit_alignment_codegen(scope, alignment, disable_reuse):
    @T.prim_func
    def kernel(src: T.Tensor((128,), "uint8"), dst: T.Tensor((144,), "uint8")):
        with T.Kernel(1, threads=32):
            a_pad = T.alloc_shared((16,), "uint8", scope=scope)
            z_shared = T.alloc_shared((128,), "uint8", scope=scope)
            for i in T.Parallel(16):
                a_pad[i] = src[i]
            T.copy(src, z_shared)
            for i in T.Parallel(128):
                dst[i] = z_shared[i]
            for i in T.Parallel(16):
                dst[128 + i] = a_pad[i]

    kernel = kernel.with_attr("tl.smem_alignment_map", {"z_shared": tvm.tirx.IntImm("int32", alignment)})
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})
    with (
        target,
        tvm.transform.PassContext(
            config={
                "tl.disable_aiu_lower": True,
                "tl.disable_shared_memory_reuse": disable_reuse,
            }
        ),
    ):
        source = tilelang.lower(kernel, target=target, enable_device_compile=False).kernel_source
    if scope == "shared":
        assert re.search(r"__align__\(16\).*\ba_pad\[", source), source
        assert re.search(rf"__align__\({alignment}\).*\bz_shared\[", source), source
    else:
        base = "buf_dyn_shmem"
        assert re.search(rf"__align__\(1024\).*\b{base}\[", source), source
        match = re.search(rf"void\*\s+z_shared\s*=.*?{base}\s*\+\s*(\d+)", source)
        assert match and int(match.group(1)) == alignment, source
