"""CPU-only checks for uniform shared alignment in AIU kernels."""

import re

import pytest
import tilelang
import tilelang.language as T
from tilelang import tvm
from tilelang.ppu.pipeline import _AlignAIUSharedMemory

pytestmark = pytest.mark.skipif(
    tvm.get_global_func("target.build.tilelang_ppu", allow_missing=True) is None,
    reason="Requires a build with PPU codegen",
)
tir = tvm.tirx


@pytest.mark.parametrize("scope,merge", [("shared", False), ("shared", True), ("shared.dyn", False)])
@pytest.mark.parametrize("use_aiu", [False, True])
def test_shared_alignment_codegen(scope, merge, use_aiu):
    @T.prim_func
    def kernel(src: T.Tensor((128,), T.uint8), dst: T.Tensor((129,), T.uint8)):
        with T.Kernel(1, threads=32):
            a_pad = T.alloc_shared((16,), T.uint8, scope=scope)
            z_shared = T.alloc_shared((128,), T.uint8, scope=scope)
            tx = T.get_thread_binding()
            if tx < 16:
                a_pad[tx] = T.cast(tx, T.uint8)
            if use_aiu:
                T.ppu_aiu_load(T.access_ptr(z_shared[0], "w"), T.access_ptr(src[0], "r"), 128, 1, 128, 1, 128, 0, 0, 0, 0)
                T.ptx_commit_group()
                T.ptx_wait_group(0)
            else:
                for i in T.Parallel(128):
                    z_shared[i] = src[i]
            T.sync_threads()
            for i in T.Parallel(128):
                dst[i] = z_shared[i]
            if tx == 0:
                dst[128] = a_pad[15]

    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})
    with target, tvm.transform.PassContext(config={"tirx.merge_static_smem": merge}):
        source = tilelang.lower(kernel, target=target, enable_device_compile=False).kernel_source
    assert ("tl::aiu_load" in source) == use_aiu
    alignment = 128 if use_aiu else 16
    if scope == "shared" and not merge:
        assert re.search(rf"__align__\({alignment}\).*\ba_pad\[16\]", source)
        assert re.search(rf"__align__\({alignment}\).*\bz_shared\[128\]", source)
    else:
        base_alignment = 1024 if scope == "shared.dyn" else alignment
        name = "buf_dyn_shmem" if scope == "shared.dyn" else "buf_shmem"
        assert re.search(rf"__align__\({base_alignment}\).*\b{name}\[", source)
        match = re.search(rf"void\*\s+z_shared\s*=.*?{name}\s*\+\s*(\d+)", source)
        assert match and int(match.group(1)) == alignment


def test_alignment_preserves_stronger_requirements():
    shared = tir.decl_buffer((128,), "uint8", name="shared", scope="shared")
    src = tir.Var("src", "handle")
    load = tir.Evaluate(tir.Call("handle", tvm.ir.Op.get("tl.ppu_aiu_load"), [shared.access_ptr("w"), src, 128, 1, 128, 1, 128, 0, 0, 0]))
    func = tir.PrimFunc([src], tir.SeqStmt([tir.AllocBuffer(shared), load]))
    func = func.with_attr("tl.smem_alignment_map", {shared.data.name: tir.IntImm("int32", 256)})
    result = _AlignAIUSharedMemory(tvm.IRModule({"main": func}))["main"]
    assert int(result.attrs["tl.smem_alignment_map"][shared.data.name]) == 256


def test_alignment_leaves_non_aiu_function_unchanged():
    shared = tir.decl_buffer((16,), "uint8", name="shared", scope="shared")
    func = tir.PrimFunc([], tir.SeqStmt([tir.AllocBuffer(shared), tir.Evaluate(shared[0])]))
    result = _AlignAIUSharedMemory(tvm.IRModule({"main": func}))["main"]
    tvm.ir.assert_structural_equal(result, func)
