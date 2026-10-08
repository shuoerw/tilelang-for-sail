"""Register ping-pong recognizes the PPU matrix-load operand contracts."""

import pytest
import tilelang
import tilelang.ppu.transform
from tilelang import tvm
from tvm import tirx as T


@pytest.mark.parametrize(
    "name",
    [
        "tl.ptx_ldmatrix",
        pytest.param(
            "tl.tix_ldmatrix_b8",
            marks=pytest.mark.skipif(
                "tl.tix_ldmatrix_b8" not in tvm.ir.Op.list_op_names(),
                reason="b8 matrix-load intrinsic is unavailable",
            ),
        ),
        "tl.tix_ldmatrix_swzl",
    ],
)
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
    assert [node.value.buffer.name for node in stores] == ["A_local_0", "A_local_1"]


def _scalar_pipeline(minimum=0, unsafe=None):
    shared = T.decl_buffer((16,), "float32", name="shared", scope="shared")
    local = T.decl_buffer((1,), "float32", name="reg", scope="local")
    output = T.decl_buffer((16,), "float32", name="output", scope="local")
    k = T.Var("k", "int32")
    load = T.BufferStore(local, T.BufferLoad(shared, [k]), [0])
    compute = T.BufferStore(output, T.BufferLoad(local, [0]) + 1, [k])
    init = None
    stmts = [load, compute]
    if unsafe == "shared_write":
        stmts[1] = T.BufferStore(shared, T.BufferLoad(local, [0]) + 1, [k + 1])
    elif unsafe == "barrier":
        stmts.insert(1, T.Evaluate(T.Call("int32", tvm.ir.Op.get("tirx.tvm_storage_sync"), ["shared"])))
    elif unsafe == "init":
        init = T.BufferStore(local, T.FloatImm("float32", 0), [0])
    elif unsafe == "load_after_compute":
        stmts = [compute, load]
    elif unsafe == "conditional":
        stmts[0] = T.IfThenElse(k < 1, load, None)
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


@pytest.mark.parametrize("unsafe", ["shared_write", "barrier", "init", "load_after_compute", "conditional"])
def test_unsafe_pipeline_keeps_original_storage_and_order(unsafe):
    result = _scalar_pipeline(unsafe=unsafe)
    assert isinstance(result.body, T.For)
    assert "register_pipeline_stage" not in result.body.annotations
    assert [b.name for b in result.alloc_buffers] == ["reg", "output"]
    if unsafe == "init":
        assert result.init.buffer.data.same_as(result.alloc_buffers[0].data)
