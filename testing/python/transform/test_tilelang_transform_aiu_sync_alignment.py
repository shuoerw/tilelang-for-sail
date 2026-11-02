"""Compiler-only checks for AIU group boundaries and pointer alignment."""

from itertools import product

import pytest
import tilelang
import tilelang.ppu.language as T
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


def _wait(count=0):
    return tir.Evaluate(tir.Call("handle", tvm.ir.Op.get("tirx.ptx_wait_group"), [count]))


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
        extent = conditions[stmt.extent.name] if isinstance(stmt.extent, tir.Var) else int(stmt.extent)
        for _ in range(extent):
            events.extend(_events(stmt.body, conditions))
    elif isinstance(stmt, tir.AttrStmt):
        events.extend(_events(stmt.body, conditions))
    elif isinstance(stmt, tir.SBlockRealize):
        enabled = conditions[stmt.predicate.name] if isinstance(stmt.predicate, tir.Var) else bool(stmt.predicate)
        if enabled:
            events.extend(_events(stmt.block, conditions))
    elif isinstance(stmt, tir.SBlock):
        if stmt.init is not None and conditions.get("run_init", True):
            events.extend(_events(stmt.init, conditions))
        events.extend(_events(stmt.body, conditions))
    elif isinstance(stmt, tir.BufferStore):
        events.append(("write", stmt.buffer.name))
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


@pytest.mark.parametrize("extent", [0, 2, "dynamic"])
@pytest.mark.parametrize("nested", [False, True])
@pytest.mark.parametrize("tail", ["reload", "different", "drained", "partial", "branch", "branch_drained"])
def test_loop_exit_preserves_new_pending_generations(extent, nested, tail):
    src = tir.Var("src", "handle")
    n = tir.Var("n", "int32")
    condition = tir.Var("condition", "bool")
    x, y, z = [tir.decl_buffer((128,), "uint8", name=name, scope="shared") for name in ("X", "Y", "Z")]
    suffix = [_load(y if tail == "different" else x, src)]
    if tail == "partial":
        suffix.extend([_load(y, src), _wait(1)])
    elif tail == "drained":
        suffix.append(_wait())
    elif tail in ("branch", "branch_drained"):
        then = tir.SeqStmt([*suffix, _wait()]) if tail == "branch_drained" else suffix[0]
        suffix = [tir.IfThenElse(condition, then, _wait())]
    body = tir.SeqStmt([_wait(), tir.Evaluate(x[0]), *suffix])
    loop = tir.For(tir.Var("i", "int32"), 0, n if extent == "dynamic" else extent, tir.ForKind.SERIAL, body)
    if nested:
        loop = tir.For(tir.Var("j", "int32"), 0, 2, tir.ForKind.SERIAL, loop)
    # Initial X is already complete. A later consumer must nevertheless wait
    # for a *new* generation issued at the loop tail, before reading X or Y.
    func = tir.PrimFunc(
        [src, n, condition, x.data, y.data, z.data],
        tir.SeqStmt([_load(x, src), _wait(), loop, _load(z, src), tir.Evaluate(x[0]), tir.Evaluate(y[0]), tir.Evaluate(z[0])]),
    )
    result = tilelang.ppu.transform.InjectAIUSyncBarrier()(tvm.IRModule({"main": func}))["main"]
    for trips, enabled in product((0, 1, 3), (False, True)):
        events = _events(result.body, {"n": trips, "condition": enabled})
        pending, open_group = [], set()
        for kind, value in events:
            if kind == "load":
                assert all(value not in group for group in pending), events
                open_group.add(value)
            elif kind == "commit":
                pending.append(open_group)
                open_group = set()
            elif kind == "wait":
                pending = pending[-value:] if value else []
            else:
                assert value not in open_group, events
                assert all(value not in group for group in pending), events
        # If all paths drain inside the loop, no wait should be inserted
        # between the final Z commit and the reads of X and Y.
        if tail in ("drained", "branch_drained") or extent == 0:
            z_load = events.index(("load", "Z"))
            assert events[z_load + 1 : z_load + 4] == [("commit", None), ("read", "X"), ("read", "Y")], events


@pytest.mark.parametrize("drain_tail", [False, True])
def test_block_exit_does_not_assume_reduction_init_always_runs(drain_tail):
    src = tir.Var("src", "handle")
    predicate = tir.Var("predicate", "bool")
    x, y = [tir.decl_buffer((128,), "uint8", name=name, scope="shared") for name in ("X", "Y")]
    body = tir.SeqStmt([_load(y, src), _wait()]) if drain_tail else tir.SeqStmt([_load(y, src), tir.Evaluate(0)])
    block = tir.SBlock([], [], [], "test", body, init=_wait())
    realize = tir.SBlockRealize([], predicate, block)
    func = tir.PrimFunc([src, predicate, x.data, y.data], tir.SeqStmt([_load(x, src), realize, tir.Evaluate(x[0]), tir.Evaluate(y[0])]))
    result = tilelang.ppu.transform.InjectAIUSyncBarrier()(tvm.IRModule({"main": func}))["main"]
    for enabled, run_init in product((False, True), repeat=2):
        events = _events(result.body, {"predicate": enabled, "run_init": run_init})
        pending, open_group = [], set()
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


@pytest.mark.parametrize("conditional_reload", [False, True])
@pytest.mark.parametrize("drain_tail", [False, True])
@pytest.mark.parametrize("nested", [False, True])
def test_prefetch_loop_preserves_incoming_dependencies(conditional_reload, drain_tail, nested):
    src = tir.Var("src", "handle")
    condition = tir.Var("condition", "bool")
    x, y, z = [tir.decl_buffer((128,), "uint8", name=name, scope="shared") for name in ("X", "Y", "Z")]
    reload = _load(x, src)
    if conditional_reload:
        reload = tir.IfThenElse(condition, reload, None)
    # Reading Y after the X prefetch drains the local queue, so no pending
    # tail remains to trigger the old wrap-around heuristic. The initial X
    # load still has to complete before the first iteration's read.
    tail = [reload, tir.Evaluate(y[0])] if drain_tail else [tir.Evaluate(y[0]), reload]
    loop = tir.For(tir.Var("i", "int32"), 0, 2, tir.ForKind.SERIAL, tir.SeqStmt([_load(y, src), tir.Evaluate(x[0]), *tail]))
    if nested:
        loop = tir.For(tir.Var("j", "int32"), 0, 2, tir.ForKind.SERIAL, loop)
    func = tir.PrimFunc([src, condition, x.data, y.data, z.data], tir.SeqStmt([_load(x, src), _load(z, src), loop, tir.Evaluate(z[0])]))
    result = tilelang.ppu.transform.InjectAIUSyncBarrier()(tvm.IRModule({"main": func}))["main"]
    for enabled in (False, True):
        events = _events(result.body, {"condition": enabled})
        # The unrelated newer Z group may remain in flight on straight-line
        # paths; conditional loads retain the pass's existing wait-all policy.
        pending, open_group = [], set()
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
        if conditional_reload and not drain_tail:
            # The generated loop-head wait-all already covers incoming X.
            assert events[4] == ("load", "Y"), events
        else:
            assert events[4] == ("wait", 0 if conditional_reload else 1), events


@pytest.mark.parametrize(
    "pattern,covered",
    [
        ("explicit", True),
        ("independent_work", True),
        ("wait_scope", True),
        ("wait_scope_one", False),
        ("opaque_scope", False),
        ("both_branches", True),
        ("guarded_use", True),
        ("one_branch", False),
        ("late_wait", False),
        ("overwrite", False),
        ("access_ptr", False),
        ("condition_read", False),
        ("bound_read", False),
        ("wait_one", False),
        ("positive_inner", True),
        ("zero_inner", False),
        ("dynamic_inner", False),
        ("dynamic_guarded_use", True),
    ],
)
@pytest.mark.parametrize("nested", [False, True])
def test_incoming_wait_coverage(pattern, covered, nested):
    src = tir.Var("src", "handle")
    condition = tir.Var("condition", "bool")
    extent = tir.Var("extent", "int32")
    x, z = [tir.decl_buffer((128,), "uint8", name=name, scope="shared") for name in ("X", "Z")]
    read = tir.Evaluate(x[0])
    body = tir.SeqStmt([_wait(), read])
    if pattern == "independent_work":
        body = tir.SeqStmt([tir.BufferStore(z, tir.IntImm("uint8", 7), [0]), _wait(), read])
    elif pattern in ("wait_scope", "wait_scope_one"):
        count = 1 if pattern == "wait_scope_one" else 0
        body = tir.AttrStmt(0, "async_wait_queue_scope", 0, tir.AttrStmt(0, "async_wait_inflight_count", count, read))
    elif pattern == "opaque_scope":
        body = tir.AttrStmt(0, "async_scope", 1, body)
    elif pattern == "both_branches":
        body = tir.SeqStmt([tir.IfThenElse(condition, _wait(), _wait()), read])
    elif pattern == "guarded_use":
        body = tir.IfThenElse(condition, body, None)
    elif pattern == "one_branch":
        body = tir.SeqStmt([tir.IfThenElse(condition, _wait(), None), read])
    elif pattern == "late_wait":
        body = tir.SeqStmt([read, _wait()])
    elif pattern == "overwrite":
        body = tir.SeqStmt([tir.BufferStore(x, tir.IntImm("uint8", 7), [0]), _wait()])
    elif pattern == "access_ptr":
        body = tir.SeqStmt([tir.Evaluate(x.access_ptr("rw")), _wait()])
    elif pattern == "condition_read":
        body = tir.IfThenElse(x[0] != 0, body, _wait())
    elif pattern == "bound_read":
        body = tir.For(tir.Var("j", "int32"), 0, tir.Cast("int32", x[0]), tir.ForKind.SERIAL, body)
    elif pattern == "wait_one":
        body = tir.SeqStmt([_wait(1), read])
    elif pattern in ("positive_inner", "zero_inner", "dynamic_inner", "dynamic_guarded_use"):
        inner_extent = 2 if pattern == "positive_inner" else 0 if pattern == "zero_inner" else extent
        inner_body = body if pattern == "dynamic_guarded_use" else _wait()
        inner = tir.For(tir.Var("j", "int32"), 0, inner_extent, tir.ForKind.SERIAL, inner_body)
        body = inner if pattern == "dynamic_guarded_use" else tir.SeqStmt([inner, read])
    loop = tir.For(tir.Var("i", "int32"), 0, 2, tir.ForKind.SERIAL, body)
    if nested:
        loop = tir.For(tir.Var("outer", "int32"), 0, 2, tir.ForKind.SERIAL, loop)
    func = tir.PrimFunc([src, condition, extent, x.data, z.data], tir.SeqStmt([_load(x, src), loop]))
    result = tilelang.ppu.transform.InjectAIUSyncBarrier()(tvm.IRModule({"main": func}))["main"]
    # Check placement, not just the total count: independent work must stay
    # before the existing wait, and a later wait cannot protect an earlier use.
    prefix = []

    def before_loop(stmt):
        if isinstance(stmt, tir.For):
            return False
        if isinstance(stmt, tir.SeqStmt):
            for child in stmt.seq:
                if not before_loop(child):
                    return False
        else:
            prefix.extend(_events(stmt, {}))
        return True

    before_loop(result.body)
    assert prefix == [("load", "X"), ("commit", None)] + ([] if covered else [("wait", 0)])
    # Buffer-dependent conditions/bounds and pointer calls are checked above
    # structurally; the simple event interpreter covers the other paths.
    if pattern not in ("condition_read", "bound_read", "access_ptr"):
        for enabled, trips in product((False, True), (0, 2)):
            pending = False
            for kind, value in _events(result.body, {"condition": enabled, "extent": trips}):
                if kind == "load" and value == "X":
                    pending = True
                elif kind == "wait" and value == 0:
                    pending = False
                elif kind in ("read", "write") and value == "X":
                    assert not pending, (pattern, enabled, trips)


@pytest.mark.parametrize("unsafe_newest", [False, True])
def test_incoming_wait_coverage_is_per_group(unsafe_newest):
    src = tir.Var("src", "handle")
    x, z = [tir.decl_buffer((128,), "uint8", name=name, scope="shared") for name in ("X", "Z")]
    first, second = (z, x) if unsafe_newest else (x, z)
    loop = tir.For(
        tir.Var("i", "int32"),
        0,
        2,
        tir.ForKind.SERIAL,
        tir.SeqStmt([tir.Evaluate(first[0]), _wait(), tir.Evaluate(second[0])]),
    )
    func = tir.PrimFunc([src, x.data, z.data], tir.SeqStmt([_load(x, src), _load(z, src), loop]))
    result = tilelang.ppu.transform.InjectAIUSyncBarrier()(tvm.IRModule({"main": func}))["main"]
    events = _events(result.body, {})
    assert events[4] == ("wait", 0 if unsafe_newest else 1), events
    assert events[5] == ("read", first.name), events


@pytest.mark.parametrize("wait_count", [0, 1])
def test_explicit_wait_all_retires_completed_groups(wait_count):
    src = tir.Var("src", "handle")
    x, z = [tir.decl_buffer((128,), "uint8", name=name, scope="shared") for name in ("X", "Z")]
    loop = tir.For(
        tir.Var("i", "int32"),
        0,
        2,
        tir.ForKind.SERIAL,
        tir.SeqStmt([tir.Evaluate(x[0]), _wait(), tir.Evaluate(z[0])]),
    )
    body = tir.SeqStmt([_load(x, src), _wait(wait_count), _load(z, src), loop])
    func = tir.PrimFunc([src, x.data, z.data], body)
    result = tilelang.ppu.transform.InjectAIUSyncBarrier()(tvm.IRModule({"main": func}))["main"]
    events = _events(result.body, {})
    assert events[5] == (("read", "X") if wait_count == 0 else ("wait", 1)), events


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
    from tilelang.ppu.pipeline import _AlignAIUSharedMemory

    mod = _AlignAIUSharedMemory(tvm.IRModule({"main": func}))
    assert int(mod["main"].attrs["tl.smem_alignment_map"][shared.data.name]) == 128
    assert int(mod["main"].attrs["tl.smem_alignment_map"][pad.data.name]) == 128
    with tvm.transform.PassContext(config={"tirx.merge_static_smem": merge}):
        result = tilelang.transform.MergeSharedMemoryAllocations(disable_reuse=True)(mod)["main"]
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
        assert int(alloc.buffer.shape[0]) == 128
        pad_alloc = next(a for a in allocations if a.buffer.data.same_as(pad.data))
        assert int(pad_alloc.buffer.shape[0]) == 16
    else:
        alias = next(b for b in bindings if b.var.same_as(shared.data))
        assert alias.value.op.name == "tirx.handle_add_byte_offset"
        offset = int(tvm.arith.Analyzer().simplify(alias.value.args[1]))
        assert offset >= 16 and offset % 128 == 0
        assert int(allocations[0].buffer.shape[0]) == 256


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
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})
    with target, tvm.transform.PassContext(config={"tl.disable_aiu_lower": False}):
        tilelang.lower(kernel, target=target, enable_device_compile=False)
    assert checked
