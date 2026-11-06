"""PPU post-placement preserves physical slots and async group dependencies."""

import pytest

import tilelang as tl
import tilelang.ppu.language as T
import tilelang.ppu.transform
from tilelang import tvm
from tvm import tirx


TARGET = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})
DEPTH = "tl_pipelined_num_stages"
pytestmark = pytest.mark.skipif(
    tvm.get_global_func("tl.ppu.transform.SchedulePipelineCopies", allow_missing=True) is None,
    reason="Requires a build with PPU transforms",
)


def _module(depth=3, iterations=8, minimum=0):
    k_extent = max(1, minimum + iterations) * 32

    @T.prim_func
    def main(A: T.Tensor((128, k_extent), T.float16), B: T.Tensor((k_extent, 128), T.float16), C: T.Tensor((128, 128), T.float32)):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((128, 32), T.float16)
            b = T.alloc_shared((32, 128), T.float16)
            c = T.alloc_fragment((128, 128), T.float32)
            T.clear(c)
            for k in T.Pipelined(minimum, minimum + iterations, num_stages=depth):
                T.copy(A[:, k * 32 : (k + 1) * 32], a)
                T.copy(B[k * 32 : (k + 1) * 32, :], b)
                T.gemm(a, b, c)
            T.copy(c, C)

    mod = tvm.IRModule.from_expr(main.with_attr("global_symbol", "main"))
    mod = tirx.transform.BindTarget(TARGET)(mod)
    mod = tl.transform.IfStmtBinding()(mod)
    mod = tl.transform.PipelinePlanning()(mod)
    return tl.transform.InjectSoftwarePipeline()(mod)


def _schedule(mod):
    return tl.ppu.transform.SchedulePipelineCopies()(mod)


def _loops(mod):
    result = []
    tirx.stmt_functor.post_order_visit(
        mod["main"].body, lambda node: result.append(node) if isinstance(node, tirx.For) and DEPTH in node.annotations else None
    )
    return result


def _events(stmt):
    if isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call):
        call = stmt.value
        if isinstance(call.op, tvm.ir.Op):
            yield call
    elif isinstance(stmt, tirx.SeqStmt):
        for child in stmt.seq:
            yield from _events(child)
    elif isinstance(stmt, tirx.SBlockRealize):
        yield from _events(stmt.block.body)


def _ops(stmt):
    return [(call.op.name, int(call.args[0]) if call.op.name == "tirx.ptx_wait_group" else None) for call in _events(stmt)]


def _allocations(mod):
    result = []

    def collect(node):
        if isinstance(node, tirx.SBlock):
            result.extend(node.alloc_buffers)
        elif isinstance(node, tirx.AllocBuffer):
            result.append(node.buffer)

    tirx.stmt_functor.post_order_visit(mod["main"].body, collect)
    return result


@pytest.mark.parametrize("depth", [2, 3, 4])
def test_copies_move_after_gemm_without_changing_storage(depth):
    before = _module(depth)
    after = _schedule(before)
    assert _ops(_loops(before)[0].body) == [
        ("tl.tileop.copy", None),
        ("tl.tileop.copy", None),
        ("tirx.ptx_commit_group", None),
        ("tirx.ptx_wait_group", depth - 1),
        ("tl.tileop.gemm", None),
    ]
    assert _ops(_loops(after)[0].body) == [
        ("tirx.ptx_wait_group", depth - 2),
        ("tl.tileop.gemm", None),
        ("tl.tileop.copy", None),
        ("tl.tileop.copy", None),
        ("tirx.ptx_commit_group", None),
    ]
    assert all(a.same_as(b) for a, b in zip(_allocations(before), _allocations(after)))
    for buffer in _allocations(after):
        if buffer.scope().startswith("shared"):
            assert int(buffer.shape[0]) == depth
    tvm.ir.assert_structural_equal(_schedule(after), after)
    assert "tl_pipeline_version" not in after.script()


def _execute_groups(mod, block_k=32, scaled=False):
    analyzer = tvm.arith.Analyzer()
    open_group, pending, memory, consumed = [], [], {}, []

    def integer(expr, bindings):
        return int(analyzer.simplify(tirx.stmt_functor.substitute(expr, bindings)))

    def visit(stmt, bindings):
        nonlocal open_group
        if isinstance(stmt, tirx.For):
            start, extent = integer(stmt.min, bindings), integer(stmt.extent, bindings)
            for i in range(start, start + extent):
                visit(stmt.body, {**bindings, stmt.loop_var: i})
        elif isinstance(stmt, tirx.SeqStmt):
            for child in stmt.seq:
                visit(child, bindings)
        elif isinstance(stmt, tirx.SBlockRealize):
            if integer(stmt.predicate, bindings):
                visit(stmt.block.body, dict(bindings))
        elif isinstance(stmt, tirx.Bind):
            if isinstance(stmt.value, tirx.Call) and stmt.value.op.name == "tl.launch_thread_idx":
                bindings[stmt.var] = 0
            else:
                bindings[stmt.var] = integer(stmt.value, bindings)
        elif isinstance(stmt, tirx.AttrStmt):
            visit(stmt.body, bindings)
        elif isinstance(stmt, tirx.IfThenElse):
            if integer(stmt.condition, bindings):
                visit(stmt.then_case, bindings)
            elif stmt.else_case is not None:
                visit(stmt.else_case, bindings)
        elif isinstance(stmt, tirx.Evaluate) and isinstance(stmt.value, tirx.Call):
            call = stmt.value
            name = call.op.name
            if name == "tl.tileop.copy":
                src, dst = call.args[0].args[0], call.args[1].args[0]
                if not dst.buffer.scope().startswith("shared"):
                    return
                if src.buffer.name in ("SFA", "SFB"):
                    axis, divisor = 0, block_k // 64
                else:
                    axis, divisor = (1 if scaled or src.buffer.name == "A" else 0), block_k
                tile = integer(src.indices[axis], bindings) // divisor
                slot = integer(dst.indices[0], bindings)
                open_group.append((dst.buffer.data, slot, tile))
            elif name == "tirx.ptx_commit_group":
                pending.append(open_group)
                open_group = []
            elif name == "tirx.ptx_wait_group":
                while len(pending) > integer(call.args[0], bindings):
                    for data, slot, tile in pending.pop(0):
                        memory[data, slot] = tile
            elif name in ("tl.tileop.gemm", "tl.tileop.gemm_blockscaled"):
                inputs = [arg.args[0] for arg in call.args[:2]]
                if scaled:
                    inputs.extend(arg.args[0] for arg in call.args[13:15])
                tiles = [memory[load.buffer.data, integer(load.indices[0], bindings)] for load in inputs]
                assert all(tile == tiles[0] for tile in tiles)
                consumed.append(tiles[0])

    visit(mod["main"].body, {})
    assert not open_group and not pending
    return consumed


@pytest.mark.parametrize("depth", [2, 3, 4])
@pytest.mark.parametrize("minimum,iterations", [(0, 0), (0, 1), (0, 2), (0, 8), (3, 7)])
def test_prologue_steady_state_and_epilogue_read_completed_groups(depth, minimum, iterations):
    before = _module(depth, iterations, minimum)
    expected = list(range(minimum, minimum + iterations))
    assert _execute_groups(before) == expected
    assert _execute_groups(_schedule(before)) == expected


@pytest.mark.parametrize("block_k", [64, 128, 192])
@pytest.mark.parametrize("depth", [2, 3])
def test_mxfp4_scale_index_bindings_move_with_their_copy(block_k, depth):
    packs = block_k // 64

    @T.prim_func
    def main(
        A: T.Tensor((128, 8 * block_k), T.float4_e2m1fn),
        B: T.Tensor((128, 8 * block_k), T.float4_e2m1fn),
        SFA: T.Tensor((8 * packs, 128), T.uint16),
        SFB: T.Tensor((8 * packs, 128), T.uint16),
        C: T.Tensor((128, 128), T.float32),
    ):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((128, block_k), T.float4_e2m1fn)
            b = T.alloc_shared((128, block_k), T.float4_e2m1fn)
            sfa = T.alloc_shared((packs, 128), T.uint16)
            sfb = T.alloc_shared((packs, 128), T.uint16)
            c = T.alloc_fragment((128, 128), T.float32)
            T.clear(c)
            for k in T.Pipelined(8, num_stages=depth):
                T.copy(A[:, k * block_k : (k + 1) * block_k], a)
                T.copy(B[:, k * block_k : (k + 1) * block_k], b)
                scale_k = k * packs
                T.copy(SFA[scale_k : scale_k + packs, :], sfa)
                T.copy(SFB[scale_k : scale_k + packs, :], sfb)
                T.gemm(a, b, c, transpose_B=True, scale_A=sfa, scale_B=sfb)
            T.copy(c, C)

    mod = tvm.IRModule.from_expr(main)
    mod = tirx.transform.BindTarget(TARGET)(mod)
    mod = tl.transform.IfStmtBinding()(mod)
    mod = tl.transform.PipelinePlanning()(mod)
    before = tl.transform.InjectSoftwarePipeline()(mod)
    after = _schedule(before)
    assert _ops(_loops(after)[0].body) == [
        ("tirx.ptx_wait_group", depth - 2),
        ("tl.tileop.gemm_blockscaled", None),
        *[("tl.tileop.copy", None)] * 4,
        ("tirx.ptx_commit_group", None),
    ]
    assert _execute_groups(before, block_k, scaled=True) == list(range(8))
    assert _execute_groups(after, block_k, scaled=True) == list(range(8))


def _mutate_steady_loop(mod, mutate):
    def replace(node):
        if isinstance(node, tirx.For) and DEPTH in node.annotations:
            body = mutate(node.body)
            return tirx.For(
                node.loop_var,
                node.min,
                node.extent,
                node.kind,
                body,
                thread_binding=node.thread_binding,
                annotations=node.annotations,
                step=node.step,
            )

    func = mod["main"]
    body = tirx.stmt_functor.ir_transform(func.body, None, replace, ["tirx.For"])
    return tvm.IRModule({"main": func.with_body(body)})


@pytest.mark.parametrize(
    "unsafe",
    [
        "unknown_call",
        "extra_commit",
        "conditional",
        "zero_wait",
        "overlap",
        "view",
        "address_read",
        "caller_copy",
        "out_of_bounds",
        "address_offset",
    ],
)
def test_unproven_schedules_are_unchanged(unsafe):
    mod = _module()

    def mutate(body):
        stmts = list(body.seq)
        if unsafe == "unknown_call":
            stmts.insert(0, tirx.Evaluate(tirx.call_extern("int32", "side_effect")))
        elif unsafe == "extra_commit":
            stmts.insert(0, stmts[2])
        elif unsafe == "conditional":
            stmts[0] = tirx.IfThenElse(tirx.Var("condition", "bool"), stmts[0], None)
        elif unsafe == "zero_wait":
            stmts[3] = tirx.Evaluate(tirx.Call("handle", tvm.ir.Op.get("tirx.ptx_wait_group"), [0]))
        else:
            future = next(_events(stmts[0])).args[1].args[0].indices[0]
            output = next(_events(stmts[-1])).args[2].args[0].buffer

            def change(node):
                if not isinstance(node, tirx.Call):
                    return
                name = node.op.name
                args = list(node.args)
                annotations = dict(node.annotations)
                if unsafe in ("overlap", "out_of_bounds") and name == "tl.tileop.gemm":
                    region = args[0]
                    load = region.args[0]
                    indices = list(load.indices)
                    if unsafe == "overlap":
                        indices[0] = future
                    else:
                        indices[-1] += load.buffer.shape[-1]
                    region_args = [tirx.BufferLoad(load.buffer, indices), *region.args[1:]]
                    args[0] = tirx.Call(region.dtype, region.op, region_args)
                elif name == "tl.tileop.copy":
                    if unsafe == "caller_copy":
                        annotations.pop("no_implicit_async_commit_wait", None)
                    elif unsafe in ("view", "address_read", "address_offset"):
                        index = 1 if unsafe == "view" else 0
                        region = args[index]
                        load = region.args[0]
                        indices = list(load.indices)
                        buffer = load.buffer
                        if unsafe == "view":
                            buffer = tirx.decl_buffer(
                                buffer.shape, buffer.dtype, name="alias", data=buffer.data, scope=buffer.scope(), elem_offset=1
                            )
                        elif unsafe == "address_offset":
                            buffer = tirx.decl_buffer(
                                buffer.shape,
                                buffer.dtype,
                                name="source_view",
                                data=buffer.data,
                                elem_offset=tirx.Cast("int32", tirx.BufferLoad(output, [0, 0])),
                            )
                        else:
                            indices[-1] += tirx.Cast("int32", tirx.BufferLoad(output, [0, 0]))
                        args[index] = tirx.Call(region.dtype, region.op, [tirx.BufferLoad(buffer, indices), *region.args[1:]])
                    else:
                        return
                else:
                    return
                return tirx.Call(node.dtype, node.op, args, annotations=annotations)

            return tirx.stmt_functor.ir_transform(body, None, change, ["tirx.Call"])
        return tirx.SeqStmt(stmts)

    mod = _mutate_steady_loop(mod, mutate)
    tvm.ir.assert_structural_equal(_schedule(mod), mod)


@pytest.mark.parametrize("dependency", [None, "memory_read", "consumer_use"])
def test_replayed_bindings_require_independence(dependency):
    mod = _module()

    def mutate(body):
        stmts = list(body.seq)
        first = stmts[0]
        copy = next(_events(first))
        region, load = copy.args[0], copy.args[0].args[0]
        bound = tirx.Var("source_column", "int32")
        value = load.indices[-1]
        if dependency == "memory_read":
            output = next(_events(stmts[-1])).args[2].args[0].buffer
            value = tirx.Cast("int32", tirx.BufferLoad(output, [0, 0]))
        indices = [*load.indices[:-1], bound]
        source = tirx.Call(region.dtype, region.op, [tirx.BufferLoad(load.buffer, indices), *region.args[1:]])
        call = tirx.Call(copy.dtype, copy.op, [source, *copy.args[1:]], annotations=copy.annotations)
        block = first.block
        block = tirx.SBlock(
            block.iter_vars,
            block.reads,
            block.writes,
            block.name_hint,
            tirx.SeqStmt([tirx.Bind(bound, value), tirx.Evaluate(call)]),
            alloc_buffers=block.alloc_buffers,
        )
        stmts[0] = tirx.SBlockRealize(first.iter_values, first.predicate, block)
        if dependency == "consumer_use":

            def change(node):
                if isinstance(node, tirx.Call) and node.op.name == "tl.tileop.gemm":
                    args = list(node.args)
                    args[9] = bound == 0
                    return tirx.Call(node.dtype, node.op, args, annotations=node.annotations)

            stmts[-1] = tirx.stmt_functor.ir_transform(stmts[-1], None, change, ["tirx.Call"])
        return tirx.SeqStmt(stmts)

    before = _mutate_steady_loop(mod, mutate)
    after = _schedule(before)
    if dependency:
        tvm.ir.assert_structural_equal(after, before)
    else:
        assert _ops(_loops(after)[0].body)[0] == ("tirx.ptx_wait_group", 1)
        assert _execute_groups(before) == _execute_groups(after) == list(range(8))


def test_non_ppu_target_is_unchanged():
    mod = _module()
    func = mod["main"].with_attr("target", tvm.target.Target("llvm"))
    mod = tvm.IRModule({"main": func})
    tvm.ir.assert_structural_equal(_schedule(mod), mod)


@pytest.mark.parametrize("placement", ["function", "block", "scope"])
@pytest.mark.parametrize("annotation", ["layout_map", "safe_value_map"])
def test_enclosing_address_annotations_are_unchanged(placement, annotation):
    mod = _module()
    buffer = next(buf for buf in _allocations(mod) if buf.scope().startswith("shared"))
    if annotation == "layout_map":
        value = tl.Layout(buffer.shape, lambda version, i, j: (i, j))
    else:
        output = next(buf for buf in _allocations(mod) if buf.scope() == "local.fragment")
        value = tirx.BufferLoad(output, [0, 0])
    mapping = {buffer.data: value}
    func = mod["main"]
    if placement == "function":
        func = func.with_attr(annotation, mapping)
    elif placement == "scope":
        func = func.with_body(tirx.AttrStmt(mapping, annotation, 0, func.body))
    else:

        def annotate(node):
            if isinstance(node, tirx.SBlock) and any(b.same_as(buffer) for b in node.alloc_buffers):
                return tirx.SBlock(
                    node.iter_vars,
                    node.reads,
                    node.writes,
                    node.name_hint,
                    node.body,
                    init=node.init,
                    alloc_buffers=node.alloc_buffers,
                    match_buffers=node.match_buffers,
                    annotations={**dict(node.annotations), annotation: mapping},
                )

        func = func.with_body(tirx.stmt_functor.ir_transform(func.body, None, annotate, ["tirx.SBlock"]))
    mod = tvm.IRModule({"main": func})
    tvm.ir.assert_structural_equal(_schedule(mod), mod)


def test_volatile_accesses_are_unchanged():
    mod = _module()
    func = mod["main"]
    source = func.buffer_map[func.params[0]]
    func = func.with_body(tirx.AttrStmt(source.data, "volatile_scope", 1, func.body))
    mod = tvm.IRModule({"main": func})
    tvm.ir.assert_structural_equal(_schedule(mod), mod)
