"""Compiler-only checks for AIU group boundaries and pointer alignment."""

from itertools import product

import pytest
import tilelang
import tilelang.language as T
import tilelang.ppu.transform
from tilelang import tvm


pytestmark = pytest.mark.skipif(
    not hasattr(tilelang.ppu.transform._ffi_api, "InjectAIUSyncBarrier"),
    reason="Requires a build with PPU transforms",
)


tir = tvm.tirx


def _load(buffer, src, caller=False, pointer=None):
    return tir.Evaluate(
        tir.Call(
            "handle",
            tvm.ir.Op.get("tl.ppu_aiu_load"),
            [buffer.access_ptr("w") if pointer is None else pointer, src, 128, 1, 128, 1, 128, 0, 0, 0],
            annotations={"no_implicit_async_commit_wait": tir.IntImm("int32", 1)} if caller else {},
        )
    )


def _commit():
    return tir.Evaluate(tir.Call("handle", tvm.ir.Op.get("tirx.ptx_commit_group"), []))


def _events(stmt, conditions):
    events = []
    if isinstance(stmt, tir.SeqStmt):
        for child in stmt.seq:
            events.extend(_events(child, conditions))
    elif isinstance(stmt, tir.IfThenElse):
        branch = stmt.then_case if conditions[stmt.condition.name] else stmt.else_case
        if branch is not None:
            events.extend(_events(branch, conditions))
    elif isinstance(stmt, tir.For):
        for _ in range(int(stmt.extent)):
            events.extend(_events(stmt.body, conditions))
    elif isinstance(stmt, tir.AttrStmt):
        events.extend(_events(stmt.body, conditions))
    elif isinstance(stmt, tir.Evaluate):
        value = stmt.value
        if isinstance(value, tir.BufferLoad):
            events.append(("read", value.buffer.name))
        elif isinstance(value, tir.Call):
            if value.op.name == "tl.ppu_aiu_load":
                events.append(("load", value.args[0].args[1].name))
            elif value.op.name == "tirx.ptx_commit_group":
                events.append(("commit", None))
            elif value.op.name == "tirx.ptx_wait_group":
                events.append(("wait", int(value.args[0])))
    return events


@pytest.mark.parametrize("pattern", ["reported", "mixed", "nested", "unequal", "optional", "loop", "reversed"])
def test_branch_commit_and_wait_safety(pattern):
    src = tir.Var("src", "handle")
    condition = tir.Var("condition", "bool")
    inner = tir.Var("inner", "bool")
    buffers = {name: tir.decl_buffer((128,), "uint8", name=name, scope="shared") for name in ("R", "X", "Y", "A", "Z")}

    def load(name, caller=False):
        return _load(buffers[name], src, caller)

    def read(name):
        return tir.Evaluate(tir.BufferLoad(buffers[name], [0]))

    then = load("X")
    otherwise = tir.SeqStmt([load("A", True), _commit()])
    if pattern == "nested":
        then = tir.IfThenElse(inner, load("X"), tir.SeqStmt([load("A", True), _commit()]))
    elif pattern == "unequal":
        then = tir.SeqStmt([load("X"), load("Z")])
        otherwise = load("X")
    elif pattern == "optional":
        otherwise = None
    if pattern == "reversed":
        then, otherwise = otherwise, then
    statements = [tir.IfThenElse(condition, then, otherwise), load("Y"), read("X"), read("Y")]
    if pattern != "reported":
        statements.insert(0, load("R"))
        statements.insert(3, read("R"))
    body = tir.SeqStmt(statements)
    if pattern == "loop":
        body = tir.For(tir.Var("i", "int32"), 0, 2, tir.ForKind.SERIAL, body)
    func = tir.PrimFunc([src, condition, inner] + [b.data for b in buffers.values()], body)
    result = tilelang.ppu.transform.InjectAIUSyncBarrier()(tvm.IRModule({"main": func}))["main"]
    for condition_value in (False, True):
        for inner_value in (False, True):
            events = _events(result.body, {"condition": condition_value, "inner": inner_value})
            if pattern == "reported" and condition_value:
                assert events == [
                    ("load", "X"),
                    ("commit", None),
                    ("load", "Y"),
                    ("commit", None),
                    ("wait", 0),
                    ("read", "X"),
                    ("read", "Y"),
                ]
            open_group = set()
            pending = []
            for kind, value in events:
                if kind == "load":
                    open_group.add(value)
                elif kind == "commit":
                    pending.append(open_group)
                    open_group = set()
                elif kind == "wait":
                    assert value == 0, events
                    pending = pending[-value:] if value else []
                else:
                    assert value not in open_group, events
                    assert all(value not in group for group in pending), events
            assert not open_group, events
            # The caller-managed load and commit remain adjacent on that path.
            for i, event in enumerate(events):
                if event == ("load", "A"):
                    assert events[i + 1] == ("commit", None), events


@pytest.mark.parametrize("arms", [2, 3, 5])
@pytest.mark.parametrize("in_loop", [False, True])
def test_multiple_branch_paths_wait_for_all_groups(arms, in_loop):
    src = tir.Var("src", "handle")
    conditions = [tir.Var(f"c{i}", "bool") for i in range(arms - 1)]
    buffers = [tir.decl_buffer((128,), "uint8", name=f"S{i}", scope="shared") for i in range(arms + 2)]
    # Each arm issues a different number of commits. Include a copy before
    # and after the branch to test both incoming and outgoing pending groups.
    branch = _load(buffers[1], src)
    for i, condition in enumerate(reversed(conditions)):
        loads = [_load(buffer, src) for buffer in buffers[1 : i + 3]]
        branch = tir.IfThenElse(condition, tir.SeqStmt(loads), branch)
    reads = [tir.Evaluate(tir.BufferLoad(buffer, [0])) for buffer in (buffers[0], buffers[1], buffers[-1])]
    body = tir.SeqStmt([_load(buffers[0], src), branch, _load(buffers[-1], src), *reads])
    if in_loop:
        body = tir.For(tir.Var("i", "int32"), 0, 2, tir.ForKind.SERIAL, body)
    func = tir.PrimFunc([src, *conditions, *[b.data for b in buffers]], body)
    result = tilelang.ppu.transform.InjectAIUSyncBarrier()(tvm.IRModule({"main": func}))["main"]
    for values in product((False, True), repeat=len(conditions)):
        events = _events(result.body, dict(zip((c.name for c in conditions), values)))
        outstanding = set()
        open_group = set()
        for kind, value in events:
            if kind == "load":
                open_group.add(value)
            elif kind == "commit":
                outstanding.update(open_group)
                open_group.clear()
            elif kind == "wait":
                assert value == 0, events
                outstanding.clear()
            else:
                assert value not in open_group | outstanding, events


def test_straight_line_loads_keep_precise_waits():
    src = tir.Var("src", "handle")
    x, y = [tir.decl_buffer((128,), "uint8", name=name, scope="shared") for name in ("X", "Y")]
    body = tir.SeqStmt([_load(x, src), _load(y, src), tir.Evaluate(x[0]), tir.Evaluate(y[0])])
    func = tir.PrimFunc([src, x.data, y.data], body)
    result = tilelang.ppu.transform.InjectAIUSyncBarrier()(tvm.IRModule({"main": func}))["main"]
    assert _events(result.body, {}) == [
        ("load", "X"),
        ("commit", None),
        ("load", "Y"),
        ("commit", None),
        ("wait", 1),
        ("read", "X"),
        ("wait", 0),
        ("read", "Y"),
    ]


@pytest.mark.parametrize("incoming", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_commit_scope_owns_mixed_loads_and_closes_incoming_group(incoming, nested):
    src = tir.Var("src", "handle")
    condition = tir.Var("condition", "bool")
    r, a, x, y = [tir.decl_buffer((128,), "uint8", name=name, scope="shared") for name in ("R", "A", "X", "Y")]
    loads = tir.SeqStmt([_load(a, src, caller=True), _load(x, src)])
    if nested:
        loads = tir.IfThenElse(condition, loads, _load(a, src, caller=True))
    scope = tir.AttrStmt(tir.IntImm("int32", 0), "async_commit_queue_scope", 0, loads)
    statements = ([_load(r, src, caller=True)] if incoming else []) + [scope, _load(y, src)]
    func = tir.PrimFunc([src, condition, r.data, a.data, x.data, y.data], tir.SeqStmt(statements))
    result = tilelang.ppu.transform.InjectAIUSyncBarrier()(tvm.IRModule({"main": func}))["main"]
    for value in (False, True):
        expected = [("load", "R")] if incoming else []
        expected += [("load", "A")]
        if not nested or value:
            expected += [("load", "X")]
        expected += [("commit", None), ("load", "Y"), ("commit", None)]
        assert _events(result.body, {"condition": value}) == expected


@pytest.mark.parametrize("pattern", ["scope_then_overlap", "partial_scope", "other_queue", "nested", "loop_carried"])
def test_overlap_validation_outside_owned_scope(pattern):
    src = tir.Var("src", "handle")
    condition = tir.Var("condition", "bool")
    a, x = [tir.decl_buffer((128,), "uint8", name=name, scope="shared") for name in ("A", "X")]
    direct, compiler = _load(a, src, caller=True), _load(x, src)
    scope = tir.AttrStmt(tir.IntImm("int32", 0), "async_commit_queue_scope", 0, direct)
    if pattern == "scope_then_overlap":
        statements = [scope, direct, compiler]
    elif pattern == "partial_scope":
        statements = [direct, tir.IfThenElse(condition, scope, None), compiler]
    elif pattern == "other_queue":
        statements = [tir.AttrStmt(tir.IntImm("int32", 0), "async_commit_queue_scope", 1, tir.SeqStmt([direct, compiler]))]
    elif pattern == "nested":
        statements = [direct, tir.IfThenElse(condition, tir.IfThenElse(condition, compiler, None), None)]
    else:
        statements = [tir.For(tir.Var("i", "int32"), 0, 2, tir.ForKind.SERIAL, tir.SeqStmt([compiler, scope, direct]))]
    body = statements[0] if len(statements) == 1 else tir.SeqStmt(statements)
    func = tir.PrimFunc([src, condition, a.data, x.data], body)
    with pytest.raises(tvm.error.InternalError, match="uncommitted caller-managed async group"):
        tilelang.ppu.transform.InjectAIUSyncBarrier()(tvm.IRModule({"main": func}))


@pytest.mark.parametrize("scope,merge", [("shared", False), ("shared", True), ("shared.dyn", False)])
@pytest.mark.parametrize("binding", ["direct", "bind", "chain", "let"])
def test_bound_aiu_destination_alignment(scope, merge, binding):
    src = tir.Var("src", "handle")
    pad = tir.decl_buffer((16,), "uint8", name="a_pad", scope=scope)
    shared = tir.decl_buffer((128,), "uint8", name="z_shared", scope=scope)
    p = tir.Var("p", "handle")
    q = tir.Var("q", "handle")
    stmts = [tir.AllocBuffer(pad), tir.AllocBuffer(shared), tir.BufferStore(pad, tir.IntImm("uint8", 1), [0])]
    ptr = shared.access_ptr("w")
    if binding in ("bind", "chain"):
        stmts.append(tir.Bind(p, ptr))
        ptr = p
        if binding == "chain":
            stmts.append(tir.Bind(q, p))
            ptr = q
    call = _load(shared, src, caller=True, pointer=ptr)
    if binding == "let":
        call = tir.Evaluate(tir.Let(p, ptr, _load(shared, src, caller=True, pointer=p).value))
    stmts += [call, _commit(), tir.Evaluate(tir.BufferLoad(pad, [0])), tir.Evaluate(tir.BufferLoad(shared, [0]))]
    tx = tvm.te.thread_axis("threadIdx.x")
    body = tir.AttrStmt(tx, "thread_extent", 32, tir.SeqStmt(stmts))
    func = tir.PrimFunc([src], body)
    with tvm.transform.PassContext(config={"tirx.merge_static_smem": merge}):
        result = tilelang.transform.MergeSharedMemoryAllocations(disable_reuse=True)(tvm.IRModule({"main": func}))["main"]
    allocations = []
    bindings = []

    def collect(node):
        if isinstance(node, tir.AllocBuffer):
            allocations.append(node)
        if isinstance(node, tir.Bind):
            bindings.append(node)

    tir.stmt_functor.post_order_visit(result.body, collect)
    if scope == "shared" and not merge:
        alloc = next(a for a in allocations if a.buffer.data.same_as(shared.data))
        assert int(alloc.annotations.get("tl.required_smem_alignment", 0)) >= 128
        pad_alloc = next(a for a in allocations if a.buffer.data.same_as(pad.data))
        assert int(pad_alloc.annotations.get("tl.required_smem_alignment", 0)) == 0
    else:
        alias = next(b for b in bindings if b.var.same_as(shared.data))
        assert alias.value.op.name == "tirx.handle_add_byte_offset"
        offset = int(tvm.arith.Analyzer().simplify(alias.value.args[1]))
        assert offset >= 16 and offset % 128 == 0
        if scope == "shared":
            assert int(allocations[0].annotations.get("tl.required_smem_alignment", 0)) >= 128


@pytest.mark.parametrize("scope", ["async_scope", "async_commit_queue_scope"])
@pytest.mark.parametrize("scope_inside_branch", [False, True])
@pytest.mark.parametrize("nested_loop", [False, True])
def test_existing_async_scope_owns_branch_commit(scope, scope_inside_branch, nested_loop):
    src = tir.Var("src", "handle")
    condition = tir.Var("condition", "bool")
    shared = tir.decl_buffer((128,), "uint8", name="X", scope="shared")
    if scope_inside_branch:
        owned = tir.AttrStmt(tir.IntImm("int32", 0), scope, 0, _load(shared, src))
        if nested_loop:
            owned = tir.For(tir.Var("i", "int32"), 0, 2, tir.ForKind.SERIAL, owned)
        body = tir.IfThenElse(condition, owned, None)
    else:
        branch = tir.IfThenElse(condition, _load(shared, src), None)
        if nested_loop:
            branch = tir.For(tir.Var("i", "int32"), 0, 2, tir.ForKind.SERIAL, branch)
        body = tir.AttrStmt(tir.IntImm("int32", 0), scope, 0, branch)
    func = tir.PrimFunc([src, condition, shared.data], body)
    result = tilelang.ppu.transform.InjectAIUSyncBarrier()(tvm.IRModule({"main": func}))["main"]
    for condition_value in (False, True):
        events = _events(result.body, {"condition": condition_value})
        expected = [("load", "X")] if condition_value else []
        if nested_loop and not scope_inside_branch:
            expected *= 2
        if scope == "async_commit_queue_scope" and (condition_value or not scope_inside_branch):
            expected.append(("commit", None))
        if nested_loop and scope_inside_branch:
            expected *= 2
        assert events == expected


@pytest.mark.parametrize("num_stages", [2, 3])
@pytest.mark.parametrize("block_m", [16, 32])
def test_pipeline_aiu_loads_retain_group_ownership(monkeypatch, num_stages, block_m):
    @T.prim_func
    def kernel(
        A: T.Tensor((block_m, 256), T.float16),
        B: T.Tensor((256, 128), T.float16),
        C: T.Tensor((block_m, 128), T.float16),
    ):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((block_m, 64), T.float16)
            b = T.alloc_shared((64, 128), T.float16)
            c = T.alloc_fragment((block_m, 128), T.float32)
            T.clear(c)
            for k in T.Pipelined(4, num_stages=num_stages):
                T.copy(A[0, k * 64], a)
                T.copy(B[k * 64, 0], b)
                T.gemm(a, b, c)
            T.copy(c, C)

    original_factory = tilelang.ppu.transform.InjectAIUSyncBarrier
    checked = []

    def checked_factory():
        run_pass = original_factory()

        def check(mod):
            loads, commits, partial_warp_guards = [], [], []

            def collect(node):
                if isinstance(node, tir.Call):
                    if node.op.name == "tl.ppu_aiu_load":
                        loads.append(node)
                    elif node.op.name == "tirx.ptx_commit_group":
                        commits.append(node)
                elif (
                    isinstance(node, tir.IfThenElse)
                    and isinstance(node.condition, tir.LT)
                    and isinstance(node.condition.b, tir.IntImm)
                    and int(node.condition.b) == 64
                ):
                    partial_warp_guards.append(node)

            for func in mod.functions.values():
                tir.stmt_functor.post_order_visit(func.body, collect)
            assert loads and commits
            if block_m == 16:
                assert partial_warp_guards, "Expected only two of four warps to load A"
            assert all(int(load.annotations.get("no_implicit_async_commit_wait", 0)) == 1 for load in loads)
            result = run_pass(mod)
            # Check exact group boundaries, predicates and waits, rather than
            # only checking the total number of commit sites.
            tvm.ir.assert_structural_equal(result, mod)
            checked.append(True)
            return result

        return check

    monkeypatch.setattr(tilelang.ppu.transform, "InjectAIUSyncBarrier", checked_factory)
    with tvm.transform.PassContext(config={"tl.disable_aiu_lower": False}):
        tilelang.lower(kernel, target={"kind": "ppu", "arch": "ppu_15"}, enable_device_compile=False)
    assert checked
