"""Shared-memory alignment metadata, independent of device execution."""

import re

import pytest
import tilelang
import tilelang.language as T
from tilelang import tvm


tir = tvm.tirx
ALIGNMENT_MAP = "tl.smem_alignment_map"


def _plan(scope, merge, instruction, explicit, binding):
    src = tir.Var("src", "handle")
    barrier = tir.Var("barrier", "handle")
    pad = tir.decl_buffer((16,), "uint8", name="a_pad", scope=scope)
    shared = tir.decl_buffer((128,), "uint8", name="z_shared", scope=scope)
    pointer = shared.access_ptr("w")
    stmts = [tir.AllocBuffer(pad), tir.AllocBuffer(shared), tir.BufferStore(pad, tir.IntImm("uint8", 1), [0])]
    if binding:
        p = tir.Var("p", "handle")
        q = tir.Var("q", "handle")
        stmts.extend([tir.Bind(p, pointer), tir.Bind(q, p)])
        pointer = q
    if instruction == "aiu":
        call = tir.Call("handle", tvm.ir.Op.get("tl.ppu_aiu_load"), [pointer, src, 128, 1, 128, 1, 128, 0, 0, 0])
    else:
        call = tir.Call("handle", tvm.ir.Op.get("tl.tma_load"), [src, barrier, pointer, 0, 0])
    stmts.extend([tir.Evaluate(call), tir.Evaluate(pad[0]), tir.Evaluate(shared[0])])
    body = tir.AttrStmt(tvm.te.thread_axis("threadIdx.x"), "thread_extent", 32, tir.SeqStmt(stmts))
    func = tir.PrimFunc([src, barrier], body)
    if explicit:
        func = func.with_attr(ALIGNMENT_MAP, {shared.data.name: tir.IntImm("int32", explicit)})
    with tvm.transform.PassContext(config={"tirx.merge_static_smem": merge}):
        result = tilelang.transform.MergeSharedMemoryAllocations(disable_reuse=True)(tvm.IRModule({"main": func}))["main"]
    return result, shared, pad


@pytest.mark.parametrize("scope,merge", [("shared", False), ("shared", True), ("shared.dyn", False)])
@pytest.mark.parametrize("instruction,explicit,expected", [("aiu", 0, 128), ("aiu", 256, 256), ("tma", 0, 1024), ("tma", 256, 256)])
@pytest.mark.parametrize("binding", [False, True])
def test_alignment_map_preserves_requirements(scope, merge, instruction, explicit, expected, binding):
    result, shared, pad = _plan(scope, merge, instruction, explicit, binding)
    alignments = result.attrs[ALIGNMENT_MAP]
    assert int(alignments[shared.data.name]) == expected
    assert pad.data.name not in alignments
    allocations, bindings = [], []

    def collect(node):
        if isinstance(node, tir.AllocBuffer):
            allocations.append(node)
        elif isinstance(node, tir.Bind):
            bindings.append(node)

    tir.stmt_functor.post_order_visit(result.body, collect)
    if scope == "shared" and not merge:
        assert sorted(int(a.buffer.shape[0]) for a in allocations) == [16, 128]
    else:
        assert len(allocations) == 1
        allocation = allocations[0]
        assert int(alignments[allocation.buffer.data.name]) == expected
        alias = next(b for b in bindings if b.var.same_as(shared.data))
        offset = int(tvm.arith.Analyzer().simplify(alias.value.args[1]))
        assert offset == expected
        assert int(allocation.buffer.shape[0]) == expected + 128


PPU_AVAILABLE = tvm.get_global_func("target.build.tilelang_ppu", allow_missing=True) is not None


@pytest.mark.skipif(not PPU_AVAILABLE, reason="Requires a build with PPU codegen")
@pytest.mark.parametrize("scope,merge", [("shared", False), ("shared", True), ("shared.dyn", False)])
@pytest.mark.parametrize("alignment", [128, 256, 2048])
def test_codegen_consumes_existing_alignment_map(scope, merge, alignment):
    @T.prim_func
    def kernel(src: T.Tensor((128,), T.uint8), dst: T.Tensor((129,), T.uint8)):
        with T.Kernel(1, threads=32):
            a_pad = T.alloc_shared((16,), T.uint8, scope=scope)
            z_shared = T.alloc_shared((128,), T.uint8, scope=scope)
            tx = T.get_thread_binding()
            if tx < 16:
                a_pad[tx] = T.cast(tx, T.uint8)
            T.sync_threads()
            T.ppu_aiu_load(T.access_ptr(z_shared[0], "w"), T.access_ptr(src[0], "r"), 128, 1, 128, 1, 128, 0, 0, 0, 0)
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            T.sync_threads()
            for i in T.Parallel(128):
                dst[i] = z_shared[i]
            if tx == 0:
                dst[128] = a_pad[15]

    kernel = kernel.with_attr(ALIGNMENT_MAP, {"z_shared": tir.IntImm("int32", alignment)})
    with tvm.transform.PassContext(config={"tirx.merge_static_smem": merge}):
        source = tilelang.lower(kernel, target={"kind": "ppu", "arch": "ppu_15"}, enable_device_compile=False).kernel_source
    if scope == "shared" and not merge:
        assert re.search(rf"__align__\({alignment}\).*\bz_shared\[128\]", source)
        assert re.search(r"__align__\(16\).*\ba_pad\[16\]", source)
    else:
        minimum = max(1024, alignment) if scope == "shared.dyn" else alignment
        name = "buf_dyn_shmem" if scope == "shared.dyn" else "buf_shmem"
        assert re.search(rf"__align__\({minimum}\).*\b{name}\[", source)
        match = re.search(rf"void\*\s+z_shared\s*=.*?{name}\s*\+\s*(\d+)", source)
        assert match and int(match.group(1)) % alignment == 0
