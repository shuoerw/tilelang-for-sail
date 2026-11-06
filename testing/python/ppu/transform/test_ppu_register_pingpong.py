"""Register ping-pong recognizes the PPU matrix-load operand contracts."""

import pytest
import tilelang
import tilelang.ppu.transform
from tilelang import tvm
from tvm import tirx as T


pytestmark = pytest.mark.skipif(
    tvm.get_global_func("tl.ppu.transform.RegisterPingPongPipeline", allow_missing=True) is None,
    reason="Requires a build with PPU transforms",
)


@pytest.mark.parametrize("name", ["tl.ptx_ldmatrix", "tl.tix_ldmatrix_b8", "tl.tix_ldmatrix_swzl"])
def test_matrix_load_destinations_are_double_buffered(name):
    shared = T.decl_buffer((1024,), "float16", name="shared", scope="shared")
    local = T.decl_buffer((8,), "float16", name="A_local", scope="local")
    output = T.decl_buffer((2,), "float16", name="output", scope="local")
    ki = T.Var("ki", "int32")

    def ptr(buffer, index, mask):
        return T.Call("handle", tvm.ir.Op.get("tl.access_ptr"), [T.BufferLoad(buffer, [index]), 8, mask])

    args = [T.IntImm("bool", 0), T.IntImm("int32", 4), ptr(shared, ki * 256, 1), ptr(local, 0, 2)]
    if name.endswith("swzl"):
        args += [T.IntImm("int32", 0), T.IntImm("int32", 0)]
    load = T.Evaluate(T.Call("handle", tvm.ir.Op.get(name), args))
    consume = T.BufferStore(output, T.BufferLoad(local, [0]), [ki])
    loop = T.For(ki, 0, 2, T.ForKind.SERIAL, T.SeqStmt([load, consume]), annotations={"register_pipeline_stage": 2})
    block = T.SBlock([], [], [], "gemm", loop, alloc_buffers=[local, output])
    func = T.PrimFunc([], T.SBlockRealize([], True, block))
    mod = tilelang.ppu.transform.RegisterPingPongPipeline()(tvm.IRModule({"main": func}))
    result = mod["main"].body.block
    assert [buf.name for buf in result.alloc_buffers] == ["A_local_0", "A_local_1", "output"]
    stores = []
    T.stmt_functor.post_order_visit(result.body, lambda node: stores.append(node) if isinstance(node, T.BufferStore) else None)
    # Views retain their metadata; the data Var identifies the physical slot.
    assert len(stores) == 2
    for node, slot in zip(stores, result.alloc_buffers[:2]):
        assert node.value.buffer.data.same_as(slot.data)
        assert node.value.buffer.dtype == local.dtype
        assert tvm.ir.structural_equal(node.value.buffer.shape, local.shape)


def _scalar_pipeline(minimum=0, unsafe=None):
    shared = T.decl_buffer((16,), "float32", name="shared", scope="shared")
    local = T.decl_buffer((1,), "float32", name="reg", scope="local")
    output = T.decl_buffer((16,), "float32", name="output", scope="local")
    k = T.Var("k", "int32")
    load = T.BufferStore(local, T.BufferLoad(shared, [k]), [0])
    compute = T.BufferStore(output, T.BufferLoad(local, [0]) + 1, [k])
    init = None
    stmts = [load, compute]
    if unsafe in ("shared_write", "alias_write"):
        if unsafe == "alias_write":
            shared = T.decl_buffer((16,), "float32", name="alias", scope="shared", data=shared.data)
        stmts[1] = T.BufferStore(shared, T.BufferLoad(local, [0]) + 1, [k + 1])
    elif unsafe == "barrier":
        stmts.insert(1, T.Evaluate(T.Call("int32", tvm.ir.Op.get("tirx.tvm_storage_sync"), ["shared"])))
    elif unsafe == "init":
        init = T.BufferStore(local, T.FloatImm("float32", 0), [0])
    elif unsafe == "load_after_compute":
        stmts = [compute, load]
    elif unsafe == "conditional":
        stmts[0] = T.IfThenElse(k < 1, load, None)
    elif unsafe == "unknown_call":
        stmts.insert(1, T.Evaluate(T.call_extern("int32", "unknown_write", shared.data)))
    loop = T.For(k, minimum, 2, T.ForKind.SERIAL, T.SeqStmt(stmts), annotations={"register_pipeline_stage": 2})
    block = T.SBlock([], [], [], "pipeline", loop, init=init, alloc_buffers=[local, output])
    func = T.PrimFunc([shared.data], T.SBlockRealize([], True, block))
    return tilelang.ppu.transform.RegisterPingPongPipeline()(tvm.IRModule({"main": func}))["main"].body.block


def test_nonzero_loop_minimum_is_preserved():
    result = _scalar_pipeline(minimum=2)
    reads = []
    T.stmt_functor.post_order_visit(
        result.body, lambda n: reads.append(int(n.indices[0])) if isinstance(n, T.BufferLoad) and n.buffer.name == "shared" else None
    )
    assert reads == [2, 3]
    assert [b.name for b in result.alloc_buffers] == ["reg_0", "reg_1", "output"]


@pytest.mark.parametrize("unsafe", ["shared_write", "alias_write", "barrier", "init", "load_after_compute", "conditional", "unknown_call"])
def test_unsafe_pipeline_keeps_original_storage_and_order(unsafe):
    result = _scalar_pipeline(unsafe=unsafe)
    assert isinstance(result.body, T.For)
    assert "register_pipeline_stage" not in result.body.annotations
    assert [b.name for b in result.alloc_buffers] == ["reg", "output"]
    if unsafe == "init":
        assert result.init.buffer.data.same_as(result.alloc_buffers[0].data)


def test_pipeline_preserves_typed_strided_view_reads_and_writes():
    shared = T.decl_buffer((2,), "uint16", name="shared", scope="shared")
    storage = T.decl_buffer((4,), "uint32", name="reg", scope="local")
    view = T.decl_buffer((2, 2), "uint16", name="view", scope="local", data=storage.data, strides=[4, 1], elem_offset=1)
    output = T.decl_buffer((2,), "uint16", name="output", scope="local")
    k = T.Var("k", "int32")
    indices = [T.IntImm("int32", 0), T.IntImm("int32", 1)]
    load = T.BufferStore(view, T.BufferLoad(shared, [k]), indices)
    consume = T.BufferStore(output, T.BufferLoad(view, indices), [k])
    loop = T.For(k, 0, 2, T.ForKind.SERIAL, T.SeqStmt([load, consume]), annotations={"register_pipeline_stage": 2})
    block = T.SBlock([], [], [], "pipeline", loop, alloc_buffers=[storage, output])
    func = T.PrimFunc([shared.data], T.SBlockRealize([], True, block))
    result = tilelang.ppu.transform.RegisterPingPongPipeline()(tvm.IRModule({"main": func}))["main"].body.block
    assert [b.name for b in result.alloc_buffers] == ["reg_0", "reg_1", "output"]
    reads, writes = [], []

    def collect(node):
        if isinstance(node, T.BufferLoad) and node.buffer.name == "view":
            reads.append(node)
        elif isinstance(node, T.BufferStore) and node.buffer.name == "view":
            writes.append(node)

    T.stmt_functor.post_order_visit(result.body, collect)
    assert len(reads) == len(writes) == 2
    for accesses in (reads, writes):
        for access, slot in zip(accesses, result.alloc_buffers[:2]):
            assert access.buffer.data.same_as(slot.data)
            assert access.buffer.dtype == view.dtype
            assert tvm.ir.structural_equal(access.buffer.shape, view.shape)
            assert tvm.ir.structural_equal(access.buffer.strides, view.strides)
            assert tvm.ir.structural_equal(access.buffer.elem_offset, view.elem_offset)
            assert tvm.ir.structural_equal(access.indices, indices)


@pytest.mark.parametrize("outside_use", [None, "read", "write", "pointer", "declaration"])
@pytest.mark.parametrize("position", ["prefix", "suffix"])
def test_pipeline_preserves_surrounding_statements_and_storage(outside_use, position):
    shared = T.decl_buffer((2,), "float32", name="shared", scope="shared")
    reg = T.decl_buffer((1,), "float32", name="reg", scope="local")
    view = T.decl_buffer((1,), "float32", name="view", scope="local", data=reg.data)
    output = T.decl_buffer((2,), "float32", name="output", scope="local")
    k = T.Var("k", "int32")
    load = T.BufferStore(reg, T.BufferLoad(shared, [k]), [0])
    compute = T.BufferStore(output, T.BufferLoad(reg, [0]), [k])
    loop = T.For(k, 0, 2, T.ForKind.SERIAL, T.SeqStmt([load, compute]), annotations={"register_pipeline_stage": 2})
    surrounding = T.BufferStore(output, T.FloatImm("float32", 0), [0])
    if outside_use == "read":
        surrounding = T.BufferStore(output, T.BufferLoad(view, [0]), [0])
    elif outside_use == "write":
        surrounding = T.BufferStore(view, T.FloatImm("float32", 0), [0])
    elif outside_use == "pointer":
        surrounding = T.Evaluate(T.call_extern("int32", "escape", reg.data))
    elif outside_use == "declaration":
        surrounding = T.DeclBuffer(view)
    body = T.SeqStmt([surrounding, loop] if position == "prefix" else [loop, surrounding])
    block = T.SBlock([], [], [], "gemm", body, alloc_buffers=[reg, output], annotations={"test_metadata": 1})
    func = T.PrimFunc([shared.data], T.SBlockRealize([], True, block))
    result = tilelang.ppu.transform.RegisterPingPongPipeline()(tvm.IRModule({"main": func}))["main"].body.block
    assert result.body.seq[0 if position == "prefix" else -1].same_as(surrounding)
    assert result.annotations["test_metadata"] == 1
    names = [buf.name for buf in result.alloc_buffers]
    if outside_use is None:
        assert names == ["reg_0", "reg_1", "output"]
        assert "register_pipeline_stage" not in result.script()
        assert "for k" not in result.script()
    else:
        assert names == ["reg", "output"]
        retained = result.body.seq[1 if position == "prefix" else 0]
        assert isinstance(retained, T.For)
        assert tvm.ir.structural_equal(retained.body, loop.body)


@pytest.mark.parametrize("dependency", [None, "destination_read", "destination_pointer_read", "source_write"])
def test_register_setup_before_load_requires_independence(dependency):
    shared = T.decl_buffer((2,), "float32", name="shared", scope="shared")
    a = T.decl_buffer((2,), "float32", name="A", scope="local")
    prepared = T.decl_buffer((1,), "float32", name="prepared", scope="local")
    b = T.decl_buffer((1,), "float32", name="B", scope="local")
    output = T.decl_buffer((2,), "float32", name="output", scope="local")
    k = T.Var("k", "int32")
    # A pure register operation remains pure when its arguments read a fragment.
    setup = T.BufferStore(prepared, T.call_pure_extern("float32", "prepare_operand", T.BufferLoad(a, [k])), [0])
    if dependency == "destination_read":
        setup = T.BufferStore(prepared, T.BufferLoad(b, [0]), [0])
    elif dependency == "destination_pointer_read":
        setup = T.BufferStore(prepared, T.call_pure_extern("float32", "read_operand", b.data), [0])
    elif dependency == "source_write":
        setup = T.BufferStore(shared, T.BufferLoad(a, [k]), [k])
    load = T.BufferStore(b, T.BufferLoad(shared, [k]), [0])
    compute = T.BufferStore(output, T.BufferLoad(prepared, [0]) + T.BufferLoad(b, [0]), [k])
    loop = T.For(k, 0, 2, T.ForKind.UNROLLED, T.SeqStmt([setup, load, compute]), annotations={"register_pipeline_stage": 2})
    block = T.SBlock([], [], [], "gemm", loop, alloc_buffers=[prepared, b, output])
    func = T.PrimFunc([shared.data, a.data], T.SBlockRealize([], True, block))
    result = tilelang.ppu.transform.RegisterPingPongPipeline()(tvm.IRModule({"main": func}))["main"].body.block
    if dependency:
        assert isinstance(result.body, T.For)
        assert tvm.ir.structural_equal(result.body.body, loop.body)
        assert [buf.name for buf in result.alloc_buffers] == ["prepared", "B", "output"]
    else:
        assert [buf.name for buf in result.alloc_buffers] == ["prepared", "B_0", "B_1", "output"]
        stores = []
        T.stmt_functor.post_order_visit(result.body, lambda n: stores.append(n) if isinstance(n, T.BufferStore) else None)
        assert [s.buffer.data.name for s in stores] == ["B_0", "B_1", "prepared", "output", "prepared", "output"]
