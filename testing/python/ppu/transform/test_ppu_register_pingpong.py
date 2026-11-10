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

    num = 2 if name == "tl.tix_ldmatrix_b8" else 4
    args = [T.IntImm("bool", 0), T.IntImm("int32", num), ptr(shared, ki * 256, 1), ptr(local, 0, 2)]
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
    elif unsafe == "allocation":
        scratch = T.decl_buffer((16,), "float32", name="scratch", scope="shared")
        stmts = [T.AllocBuffer(scratch), T.BufferStore(local, T.BufferLoad(scratch, [k]), [0]), compute]
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


@pytest.mark.parametrize(
    "unsafe", ["shared_write", "alias_write", "barrier", "init", "load_after_compute", "conditional", "unknown_call", "allocation"]
)
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


@pytest.mark.parametrize(
    "outside_use",
    [
        None,
        "read",
        "write",
        "pointer",
        "declaration",
        "descriptor_offset",
        "descriptor_shape",
        "descriptor_stride",
        "descriptor_decl",
        "descriptor_nested",
    ],
)
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
    elif outside_use and outside_use.startswith("descriptor_"):
        reference = T.Cast("int32", T.BufferLoad(reg, [0]))
        options = {"elem_offset": reference}
        shape = (1,)
        if outside_use == "descriptor_shape":
            options = {}
            shape = (reference,)
        elif outside_use == "descriptor_stride":
            options = {"strides": [reference]}
        elif outside_use == "descriptor_nested":
            inner = T.decl_buffer((1,), "float32", name="inner", scope="local", data=output.data, elem_offset=reference)
            options = {"elem_offset": T.Cast("int32", T.BufferLoad(inner, [0]))}
        descriptor = T.decl_buffer(shape, "float32", name="descriptor", scope="local", data=output.data, **options)
        surrounding = (
            T.DeclBuffer(descriptor) if outside_use == "descriptor_decl" else T.BufferStore(descriptor, T.FloatImm("float32", 0), [0])
        )
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


def _rewrite_loop(loop, shared, reg, output):
    block = T.SBlock([], [], [], "gemm", loop, alloc_buffers=[reg, output])
    func = T.PrimFunc([shared.data], T.SBlockRealize([], True, block))
    before = tvm.IRModule({"main": func})
    assert T.analysis.verify_ssa(func)
    after = tilelang.ppu.transform.RegisterPingPongPipeline()(before)
    assert T.analysis.verify_ssa(after["main"])
    return before, after


@pytest.mark.parametrize("pointer", ["raw", "nested", "access_ptr"])
def test_all_accepted_pointer_expressions_use_slot_storage(pointer):
    shared = T.decl_buffer((2,), "float32", name="shared", scope="shared")
    reg = T.decl_buffer((1,), "float32", name="reg", scope="local")
    output = T.decl_buffer((2,), "float32", name="output", scope="local")
    k = T.Var("k", "int32")
    load = T.BufferStore(reg, T.BufferLoad(shared, [k]), [0])
    argument = reg.data
    if pointer == "nested":
        argument = T.call_pure_extern("handle", "forward_pointer", reg.data)
    elif pointer == "access_ptr":
        argument = T.Call("handle", tvm.ir.Op.get("tl.access_ptr"), [T.BufferLoad(reg, [0]), 1, 1])
    compute = T.BufferStore(output, T.call_pure_extern("float32", "read_register", argument), [k])
    loop = T.For(k, 0, 2, T.ForKind.SERIAL, T.SeqStmt([load, compute]), annotations={"register_pipeline_stage": 2})
    _, after = _rewrite_loop(loop, shared, reg, output)
    result = after["main"].body.block
    assert [buf.name for buf in result.alloc_buffers] == ["reg_0", "reg_1", "output"]
    consumers, old_references = [], []

    def collect(node):
        if isinstance(node, T.Var) and node.same_as(reg.data):
            old_references.append(node)
        if isinstance(node, T.BufferStore) and node.buffer.data.same_as(output.data):
            consumers.append(node.value)

    T.stmt_functor.post_order_visit(result.body, collect)
    assert not old_references
    assert len(consumers) == 2
    for expr, slot in zip(consumers, result.alloc_buffers[:2]):
        references = []
        T.stmt_functor.post_order_visit(
            expr,
            lambda n, references=references, slot=slot: (
                references.append(n)
                if (isinstance(n, T.Var) and n.same_as(slot.data)) or (isinstance(n, T.BufferLoad) and n.buffer.data.same_as(slot.data))
                else None
            ),
        )
        assert references


def _execute_scalar(func, shared, output):
    memory = {shared.data: [10, 20, 30, 40]}
    for buffer in func.body.block.alloc_buffers:
        memory[buffer.data] = [None] * int(buffer.shape[0])
    analyzer = tvm.arith.Analyzer()

    def value(expr, bindings):
        if isinstance(expr, T.BufferLoad):
            result = memory[expr.buffer.data][value(expr.indices[0], bindings)]
            assert result is not None
            return result
        return int(analyzer.simplify(T.stmt_functor.substitute(expr, bindings)))

    def execute(stmt, bindings):
        if isinstance(stmt, T.SeqStmt):
            for child in stmt.seq:
                execute(child, bindings)
        elif isinstance(stmt, T.For):
            start, extent = value(stmt.min, bindings), value(stmt.extent, bindings)
            for i in range(start, start + extent):
                execute(stmt.body, {**bindings, stmt.loop_var: i})
        elif isinstance(stmt, T.BufferStore):
            memory[stmt.buffer.data][value(stmt.indices[0], bindings)] = value(stmt.value, bindings)
        else:
            raise AssertionError(type(stmt))

    execute(func.body.block.body, {})
    return memory[output.data]


@pytest.mark.parametrize("carried", [False, True])
def test_register_values_preserve_iteration_semantics(carried):
    shared = T.decl_buffer((4,), "float32", name="shared", scope="shared")
    reg = T.decl_buffer((2,), "float32", name="reg", scope="local")
    output = T.decl_buffer((4,), "float32", name="output", scope="local")
    k = T.Var("k", "int32")
    load = T.BufferStore(reg, T.BufferLoad(shared, [k]), [k // 2 if carried else 0])
    compute = T.BufferStore(output, T.BufferLoad(reg, [0]), [k])
    loop = T.For(k, 0, 4, T.ForKind.SERIAL, T.SeqStmt([load, compute]), annotations={"register_pipeline_stage": 2})
    before, after = _rewrite_loop(loop, shared, reg, output)
    expected = [10, 20, 20, 20] if carried else [10, 20, 30, 40]
    assert _execute_scalar(before["main"], shared, output) == expected
    assert _execute_scalar(after["main"], shared, output) == expected
    assert isinstance(after["main"].body.block.body, T.For) == carried


@pytest.mark.parametrize("consumer", ["element", "raw_pointer", "access_ptr", "typed_view", "overclaimed_load"])
def test_matrix_load_requires_coverage_of_consumed_storage(consumer):
    shared = T.decl_buffer((1024,), "float16", name="shared", scope="shared")
    reg = T.decl_buffer((16,), "float16", name="reg", scope="local")
    output = T.decl_buffer((2,), "float16", name="output", scope="local")
    k = T.Var("k", "int32")

    def ptr(buf, index, count, mask):
        return T.Call("handle", tvm.ir.Op.get("tl.access_ptr"), [T.BufferLoad(buf, [index]), count, mask])

    count = 16 if consumer == "overclaimed_load" else 8
    load = T.Evaluate(T.Call("handle", tvm.ir.Op.get("tl.ptx_ldmatrix"), [False, 4, ptr(shared, k * 256, 8, 1), ptr(reg, 0, count, 2)]))
    value = T.BufferLoad(reg, [8])
    if consumer == "raw_pointer":
        value = T.call_pure_extern("float16", "read_register", reg.data)
    elif consumer == "access_ptr":
        value = T.call_pure_extern("float16", "read_register", ptr(reg, 0, 16, 1))
    elif consumer == "typed_view":
        view = T.decl_buffer((8,), "float32", name="view", scope="local", data=reg.data)
        value = T.Cast("float16", T.BufferLoad(view, [4]))
    compute = T.BufferStore(output, value, [k])
    loop = T.For(k, 0, 2, T.ForKind.SERIAL, T.SeqStmt([load, compute]), annotations={"register_pipeline_stage": 2})
    _, after = _rewrite_loop(loop, shared, reg, output)
    result = after["main"].body.block
    assert isinstance(result.body, T.For)
    assert tvm.ir.structural_equal(result.body.body, loop.body)
    assert [b.name for b in result.alloc_buffers] == ["reg", "output"]


def test_nested_loops_have_independent_ssa_variables():
    shared = T.decl_buffer((4,), "float32", name="shared", scope="shared")
    reg = T.decl_buffer((2,), "float32", name="reg", scope="local")
    output = T.decl_buffer((4,), "float32", name="output", scope="local")
    k, i, j = [T.Var(name, "int32") for name in ("k", "i", "j")]
    load = T.For(i, 0, 2, T.ForKind.SERIAL, T.BufferStore(reg, T.BufferLoad(shared, [k * 2 + i]), [i]))
    compute = T.For(j, 0, 2, T.ForKind.SERIAL, T.BufferStore(output, T.BufferLoad(reg, [j]), [k * 2 + j]))
    loop = T.For(k, 0, 2, T.ForKind.SERIAL, T.SeqStmt([load, compute]), annotations={"register_pipeline_stage": 2})
    before, after = _rewrite_loop(loop, shared, reg, output)
    assert [b.name for b in after["main"].body.block.alloc_buffers] == ["reg_0", "reg_1", "output"]
    assert _execute_scalar(before["main"], shared, output) == [10, 20, 30, 40]
    assert _execute_scalar(after["main"], shared, output) == [10, 20, 30, 40]


def test_predicated_load_does_not_establish_iteration_local_values():
    shared = T.decl_buffer((2,), "float32", name="shared", scope="shared")
    reg = T.decl_buffer((1,), "float32", name="reg", scope="local")
    output = T.decl_buffer((2,), "float32", name="output", scope="local")
    k = T.Var("k", "int32")
    load = T.BufferStore(reg, T.BufferLoad(shared, [k]), [0], predicate=k == 0)
    compute = T.BufferStore(output, T.BufferLoad(reg, [0]), [k])
    loop = T.For(k, 0, 2, T.ForKind.SERIAL, T.SeqStmt([load, compute]), annotations={"register_pipeline_stage": 2})
    _, after = _rewrite_loop(loop, shared, reg, output)
    result = after["main"].body.block
    assert isinstance(result.body, T.For)
    assert [b.name for b in result.alloc_buffers] == ["reg", "output"]


@pytest.mark.parametrize("matrix", [False, True])
def test_mutable_address_metadata_keeps_original_schedule(matrix):
    shared = T.decl_buffer((1024,), "float32", name="shared", scope="shared")
    state = T.decl_buffer((1,), "int32", name="state", scope="local")
    source_view = T.decl_buffer((4,), "float32", name="source_view", data=shared.data, scope="shared", elem_offset=T.BufferLoad(state, [0]))
    reg = T.decl_buffer((4 if matrix else 1,), "float32", name="reg", scope="local")
    output = T.decl_buffer((2,), "float32", name="output", scope="local")
    k = T.Var("k", "int32")
    load = T.BufferStore(reg, T.BufferLoad(source_view, [0]), [0])
    if matrix:
        src = T.Call("handle", tvm.ir.Op.get("tl.access_ptr"), [T.BufferLoad(source_view, [0]), 4, 1])
        dst = T.Call("handle", tvm.ir.Op.get("tl.access_ptr"), [T.BufferLoad(reg, [0]), 4, 2])
        load = T.Evaluate(T.Call("handle", tvm.ir.Op.get("tl.ptx_ldmatrix"), [False, 4, src, dst]))
    consume = T.BufferStore(output, T.BufferLoad(reg, [0]), [k])
    advance = T.BufferStore(state, k + 1, [0])
    loop = T.For(k, 0, 2, T.ForKind.SERIAL, T.SeqStmt([load, consume, advance]), annotations={"register_pipeline_stage": 2})
    body = T.SeqStmt([T.BufferStore(state, T.IntImm("int32", 0), [0]), loop])
    block = T.SBlock([], [], [], "gemm", body, alloc_buffers=[state, reg, output])
    func = T.PrimFunc([shared.data], T.SBlockRealize([], True, block))
    result = tilelang.ppu.transform.RegisterPingPongPipeline()(tvm.IRModule({"main": func}))["main"]
    assert T.analysis.verify_ssa(result)
    assert [b.name for b in result.body.block.alloc_buffers] == ["state", "reg", "output"]
    assert isinstance(result.body.block.body.seq[-1], T.For)
    tvm.ir.assert_structural_equal(result.body.block.body.seq[-1].body, loop.body)


@pytest.mark.parametrize("pointer", ["bare", "access_ptr", "forwarded"])
def test_opaque_mma_pointer_operands_keep_dependencies(pointer):
    shared = T.decl_buffer((4096,), "float16", name="shared", scope="shared")
    reg = T.decl_buffer((8,), "float16", name="reg", scope="local")
    b = T.decl_buffer((8,), "float16", name="B", scope="local")
    accum = T.decl_buffer((8,), "float32", name="accum", scope="local")
    k, i = T.Var("k", "int32"), T.Var("i", "int32")
    index = k * 8 + i + T.Cast("int32", T.BufferLoad(accum, [0]))
    load = T.For(i, 0, 8, T.ForKind.SERIAL, T.BufferStore(reg, T.BufferLoad(shared, [index]), [i]))
    destination = accum.data
    if pointer == "access_ptr":
        destination = T.Call("handle", tvm.ir.Op.get("tl.access_ptr"), [T.BufferLoad(accum, [0]), 8, 3])
    elif pointer == "forwarded":
        destination = T.call_pure_extern("handle", "forward_accumulator", accum.data)
    args = ["m16n16k16", "row", "col", "fp16", "fp16", "fp32", reg.data, 0, b.data, 0, destination, 0, False]
    compute = T.Evaluate(T.Call("float32", tvm.ir.Op.get("tirx.ptx_mma"), args))
    loop = T.For(k, 0, 2, T.ForKind.SERIAL, T.SeqStmt([load, compute]), annotations={"register_pipeline_stage": 2})
    body = T.SeqStmt([T.BufferStore(accum, T.FloatImm("float32", 0), [0]), loop])
    block = T.SBlock([], [], [], "gemm", body, alloc_buffers=[reg, accum])
    func = T.PrimFunc([shared.data, b.data], T.SBlockRealize([], True, block))
    result = tilelang.ppu.transform.RegisterPingPongPipeline()(tvm.IRModule({"main": func}))["main"]
    retained = result.body.block.body.seq[-1]
    assert isinstance(retained, T.For)
    tvm.ir.assert_structural_equal(retained.body, loop.body)
    assert [b.name for b in result.body.block.alloc_buffers] == ["reg", "accum"]


@pytest.mark.parametrize(
    "metadata,key",
    [(metadata, key) for metadata in ["volatile_scope", "safe_function", "safe_block", "safe_scope"] for key in ["register", "shared"]]
    + [("volatile_allocation", "shared")],
)
def test_enclosing_metadata_preserves_register_schedule(metadata, key):
    shared = T.decl_buffer((2,), "float32", name="shared", scope="shared")
    reg = T.decl_buffer((1,), "float32", name="reg", scope="local")
    output = T.decl_buffer((2,), "float32", name="output", scope="local")
    k = T.Var("k", "int32")
    load = T.BufferStore(reg, T.BufferLoad(shared, [k]), [0])
    consume = T.BufferStore(output, T.BufferLoad(reg, [0]), [k])
    loop = T.For(k, 0, 2, T.ForKind.SERIAL, T.SeqStmt([load, consume]), annotations={"register_pipeline_stage": 2})
    data = reg.data if key == "register" else shared.data
    mapping = {data: T.BufferLoad(output, [0])}
    annotations = {"safe_value_map": mapping} if metadata == "safe_block" else {}
    block = T.SBlock([], [], [], "gemm", loop, alloc_buffers=[reg, output], annotations=annotations)
    body = T.SBlockRealize([], True, block)
    params = [shared.data]
    if metadata == "volatile_scope":
        body = T.AttrStmt(data, "volatile_scope", 1, body)
    elif metadata == "volatile_allocation":
        params = []
        body = T.SeqStmt([T.AllocBuffer(shared, annotations={"tirx.volatile": True}), body])
    elif metadata == "safe_scope":
        body = T.AttrStmt(mapping, "safe_value_map", 0, body)
    func = T.PrimFunc(params, body)
    if metadata == "safe_function":
        func = func.with_attr("safe_value_map", mapping)
    result = tilelang.ppu.transform.RegisterPingPongPipeline()(tvm.IRModule({"main": func}))["main"]
    loops, blocks = [], []
    T.stmt_functor.post_order_visit(
        result.body, lambda n: loops.append(n) if isinstance(n, T.For) else blocks.append(n) if isinstance(n, T.SBlock) else None
    )
    assert len(loops) == 1
    assert "register_pipeline_stage" not in loops[0].annotations
    tvm.ir.assert_structural_equal(loops[0].body, loop.body)
    assert [b.name for b in blocks[0].alloc_buffers] == ["reg", "output"]
