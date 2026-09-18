"""Compiler-only checks for AIU group boundaries and pointer alignment."""

import pytest
import tilelang
import tilelang.ppu.transform
from tilelang import tvm


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
                    ("wait", 1),
                    ("read", "X"),
                    ("wait", 0),
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
                    pending = pending[-value:] if value else []
                else:
                    assert value not in open_group, events
                    assert all(value not in group for group in pending), events
            assert not open_group, events
            # The caller-managed load and commit remain adjacent on that path.
            for i, event in enumerate(events):
                if event == ("load", "A"):
                    assert events[i + 1] == ("commit", None), events


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
def test_existing_async_scope_owns_branch_commit(scope):
    src = tir.Var("src", "handle")
    condition = tir.Var("condition", "bool")
    shared = tir.decl_buffer((128,), "uint8", name="X", scope="shared")
    branch = tir.IfThenElse(condition, _load(shared, src), None)
    body = tir.AttrStmt(tir.IntImm("int32", 0), scope, 0, branch)
    func = tir.PrimFunc([src, condition, shared.data], body)
    result = tilelang.ppu.transform.InjectAIUSyncBarrier()(tvm.IRModule({"main": func}))["main"]
    for condition_value in (False, True):
        events = _events(result.body, {"condition": condition_value})
        expected = [("load", "X")] if condition_value else []
        if scope == "async_commit_queue_scope":
            expected.append(("commit", None))
        assert events == expected
