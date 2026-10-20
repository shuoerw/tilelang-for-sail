"""Unit tests for ReorderAIULoads pass.

Verifies that aiu_load instructions are reordered by their dst buffer's
first-use position within SeqStmt blocks in stage=0 scenarios.

Scenarios covered:
  1. Basic reorder: 3 loads sorted by first-use position
  2. Dependency blocks move: address dependency prevents forward move
  3. Bundle forward move: dependency satisfied → load can move forward
  4. stage>0 skipped: For with num_stages>0 is not modified
  5. Container boundary: loads inside For are not cross-boundary reordered
  6. Slot zero assignment: second load occupies slot 0 when its use is earlier
  7. Call arg use detection: dst buffer in Call args detected as use point
"""

import tilelang  # noqa: F401 — ensures tl.ppu_aiu_load Op is registered & sets up tvm path
import tilelang.ppu.transform
import tilelang.testing
import tvm
from tvm import tirx as tir


# =========================================================================
# Helpers
# =========================================================================


def _make_access_ptr(buf):
    """Create tvm_access_ptr(placeholder, buf.data, 0, 128, 2)."""
    return tir.Call(
        "handle",
        tvm.ir.Op.get("tirx.tvm_access_ptr"),
        [
            tir.IntImm("int32", 0),  # type placeholder (pass ignores)
            buf.data,  # buffer data var (pass extracts this)
            tir.IntImm("int32", 0),  # offset
            tir.IntImm("int32", 128),  # extent
            tir.IntImm("int32", 2),  # mask (write)
        ],
    )


def _make_raw_aiu_load(buf, extra_var=None, *, compiler_managed=False):
    """Create a manual low-level Evaluate(aiu_load(...)).

    Parameters
    ----------
    buf : tir.Buffer
        The destination buffer for the aiu_load.
    extra_var : tir.Var, optional
        If provided, placed in args[1] to create an address dependency.
    """
    access = _make_access_ptr(buf)
    args = [access]
    for i in range(1, 10):
        args.append(extra_var if i == 1 and extra_var is not None else tir.IntImm("int32", 0))
    args.extend([tir.IntImm("int32", 1), tir.IntImm("int32", 128)])
    call = tir.Call(
        "handle", tir.op.Op.get("tl.ppu_aiu_load"), args,
        annotations={"no_implicit_async_commit_wait": tir.IntImm("int32", int(not compiler_managed))},
    )
    return tir.Evaluate(call)


def _make_aiu_load(buf, extra_var=None):
    """Create an AIU load whose synchronization is compiler-managed."""
    return _make_raw_aiu_load(buf, extra_var, compiler_managed=True)


def _make_use(buf):
    """Create BufferStore(buf, 0, [0]) as a 'use' of the buffer."""
    return tir.BufferStore(buf, tir.const(0, buf.dtype), [tir.IntImm("int32", 0)])


def _run_pass(body, params, buffer_map):
    """Wrap body in PrimFunc, run ReorderAIULoads, return the transformed body."""
    func = tir.PrimFunc(params, body, buffer_map=buffer_map)
    func = func.with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    mod = tilelang.ppu.transform.ReorderAIULoads()(mod)
    return mod["main"].body


def _extract_aiu_dst_names(body):
    """Extract dst buffer var names from top-level aiu_load calls in a SeqStmt.

    Only examines direct children of the SeqStmt; loads inside containers
    (For, If, Block) are NOT reported.
    """
    assert isinstance(body, tir.SeqStmt), f"Expected SeqStmt, got {type(body)}"
    names = []
    aiu_op = tir.op.Op.get("tl.ppu_aiu_load")
    for stmt in body:
        if isinstance(stmt, tir.Evaluate) and isinstance(stmt.value, tir.Call) and stmt.value.op.same_as(aiu_op):
            access = stmt.value.args[0]
            names.append(access.args[1].name)
    return names


# =========================================================================
# Tests
# =========================================================================


def test_basic_reorder():
    """3 loads (A, B, C) with use order B→A→C → reorder loads to B, A, C."""
    A = tir.decl_buffer((128,), "float16", name="A")
    B = tir.decl_buffer((128,), "float16", name="B")
    C = tir.decl_buffer((128,), "float16", name="C")

    body = tir.SeqStmt(
        [
            _make_aiu_load(A),  # slot 0
            _make_aiu_load(B),  # slot 1
            _make_aiu_load(C),  # slot 2
            _make_use(B),  # pos 3: first use of B
            _make_use(A),  # pos 4: first use of A
            _make_use(C),  # pos 5: first use of C
        ]
    )

    params = [A.data, B.data, C.data]
    buf_map = {A.data: A, B.data: B, C.data: C}
    result = _run_pass(body, params, buf_map)

    order = _extract_aiu_dst_names(result)
    assert order == ["B", "A", "C"], f"Expected ['B', 'A', 'C'], got {order}"


def test_dependency_blocks_move():
    """Load B depends on addr_var defined after load A → B cannot move before A."""
    A = tir.decl_buffer((128,), "float16", name="A")
    B = tir.decl_buffer((128,), "float16", name="B")

    addr_var = tir.Var("addr_var", "int32")
    # LetStmt defines addr_var at position 1; body is a trivial nop.
    let_stmt = tir.SeqStmt(
        [
            tir.Bind(addr_var, tir.IntImm("int32", 42)),
            tir.Evaluate(tir.IntImm("int32", 0)),
        ]
    )

    body = tir.SeqStmt(
        [
            _make_aiu_load(A),  # slot 0: load A (no deps)
            let_stmt,  # pos 1: defines addr_var
            _make_aiu_load(B, extra_var=addr_var),  # slot 2: load B (needs addr_var)
            _make_use(B),  # pos 3: first use B
            _make_use(A),  # pos 4: first use A
        ]
    )

    params = [A.data, B.data]
    buf_map = {A.data: A, B.data: B}
    result = _run_pass(body, params, buf_map)

    order = _extract_aiu_dst_names(result)
    # B has earlier first_use (3 < 4), but addr_var (def at pos 1) blocks
    # B from moving to slot 0. Order remains A, B.
    assert order == ["A", "B"], f"Expected ['A', 'B'], got {order}"


def test_bundle_forward_move():
    """Load B depends on addr_var defined before all loads → B can move forward."""
    A = tir.decl_buffer((128,), "float16", name="A")
    B = tir.decl_buffer((128,), "float16", name="B")

    addr_var = tir.Var("addr_var", "int32")
    # LetStmt defines addr_var at position 0, before both loads.
    let_stmt = tir.SeqStmt(
        [
            tir.Bind(addr_var, tir.IntImm("int32", 42)),
            tir.Evaluate(tir.IntImm("int32", 0)),
        ]
    )

    body = tir.SeqStmt(
        [
            let_stmt,  # pos 0: defines addr_var
            _make_aiu_load(A),  # slot 1: load A (no deps)
            _make_aiu_load(B, extra_var=addr_var),  # slot 2: load B (needs addr_var @ pos 0)
            _make_use(B),  # pos 3: first use B
            _make_use(A),  # pos 4: first use A
        ]
    )

    params = [A.data, B.data]
    buf_map = {A.data: A, B.data: B}
    result = _run_pass(body, params, buf_map)

    order = _extract_aiu_dst_names(result)
    # B has first_use=3, A has first_use=4. addr_var is defined at pos 0
    # which is < slot 1, so B can move to slot 1. A goes to slot 2.
    assert order == ["B", "A"], f"Expected ['B', 'A'], got {order}"


def test_stage_gt0_skipped():
    """For loop with num_stages=2 annotation → pass does NOT modify loop body."""
    A = tir.decl_buffer((128,), "float16", name="A")
    B = tir.decl_buffer((128,), "float16", name="B")

    loop_body = tir.SeqStmt(
        [
            _make_aiu_load(A),
            _make_aiu_load(B),
            _make_use(B),  # B used first
            _make_use(A),
        ]
    )

    loop_var = tir.Var("k", "int32")
    for_stmt = tir.For(
        loop_var,
        tir.IntImm("int32", 0),
        tir.IntImm("int32", 8),
        tir.ForKind.SERIAL,
        loop_body,
        annotations={"num_stages": tir.IntImm("int32", 2)},
    )

    params = [A.data, B.data]
    buf_map = {A.data: A, B.data: B}
    result = _run_pass(for_stmt, params, buf_map)

    # The For body should be unchanged (pass skips num_stages > 0)
    assert isinstance(result, tir.For), f"Expected For, got {type(result)}"
    inner_order = _extract_aiu_dst_names(result.body)
    assert inner_order == ["A", "B"], f"Expected ['A', 'B'] (unchanged), got {inner_order}"


def test_container_boundary():
    """Loads inside a For container are NOT treated as top-level loads.

    The loop is a boundary: top-level loads cannot move across it.
    """
    A = tir.decl_buffer((128,), "float16", name="A")
    B = tir.decl_buffer((128,), "float16", name="B")
    C = tir.decl_buffer((128,), "float16", name="C")

    # B's aiu_load is inside a For loop (container node)
    loop_var = tir.Var("j", "int32")
    inner_for = tir.For(
        loop_var,
        tir.IntImm("int32", 0),
        tir.IntImm("int32", 4),
        tir.ForKind.SERIAL,
        _make_aiu_load(B),
    )

    body = tir.SeqStmt(
        [
            _make_aiu_load(A),  # top-level load A (slot 0)
            inner_for,  # container with load B (NOT a top-level load)
            _make_aiu_load(C),  # top-level load C (slot 2)
            _make_use(C),  # pos 3: first use C
            _make_use(A),  # pos 4: first use A
        ]
    )

    params = [A.data, B.data, C.data]
    buf_map = {A.data: A, B.data: B, C.data: C}
    result = _run_pass(body, params, buf_map)

    # The loop may contain memory dependencies or synchronization.
    order = _extract_aiu_dst_names(result)
    assert order == ["A", "C"], f"Expected ['A', 'C'], got {order}"

    # Verify the For loop remains at position 1
    assert isinstance(result[1], tir.For), "For loop should remain at position 1"


def test_slot_zero_assignment():
    """Verify the second load occupies slot 0 when its first-use is earlier."""
    A = tir.decl_buffer((128,), "float16", name="A")
    B = tir.decl_buffer((128,), "float16", name="B")

    body = tir.SeqStmt(
        [
            _make_aiu_load(A),  # slot 0: load A
            _make_aiu_load(B),  # slot 1: load B
            _make_use(B),  # pos 2: first use of B (immediately after loads)
            tir.Evaluate(tir.IntImm("int32", 0)),  # some compute
            tir.Evaluate(tir.IntImm("int32", 0)),  # some compute
            _make_use(A),  # pos 5: first use of A (far away)
        ]
    )

    params = [A.data, B.data]
    buf_map = {A.data: A, B.data: B}
    result = _run_pass(body, params, buf_map)

    order = _extract_aiu_dst_names(result)
    assert order == ["B", "A"], f"Expected ['B', 'A'], got {order}"


def test_call_arg_use_detection():
    """Verify dst buffer passed via Call args (not tvm_access_ptr) is detected as use."""
    A = tir.decl_buffer((128,), "float16", name="A")
    B = tir.decl_buffer((128,), "float16", name="B")

    # B's use is a normal BufferStore
    use_B = _make_use(B)
    # A's use is a call_extern with A.data in its args
    call_using_A = tir.Evaluate(tir.call_extern("handle", "some_func", A.data))

    body = tir.SeqStmt(
        [
            _make_aiu_load(A),  # slot 0: load A
            _make_aiu_load(B),  # slot 1: load B
            use_B,  # pos 2: first use of B
            call_using_A,  # pos 3: first use of A (via Call arg)
        ]
    )

    params = [A.data, B.data]
    buf_map = {A.data: A, B.data: B}
    result = _run_pass(body, params, buf_map)

    order = _extract_aiu_dst_names(result)
    assert order == ["B", "A"], f"Expected ['B', 'A'], got {order}"


def test_manual_aiu_load_is_not_reordered():
    """Caller-managed AIU loads retain their explicit instruction order."""
    A = tir.decl_buffer((128,), "float16", name="A")
    B = tir.decl_buffer((128,), "float16", name="B")

    body = tir.SeqStmt(
        [
            _make_raw_aiu_load(A),
            _make_raw_aiu_load(B),
            _make_use(B),
            _make_use(A),
        ]
    )

    params = [A.data, B.data]
    buf_map = {A.data: A, B.data: B}
    result = _run_pass(body, params, buf_map)

    order = _extract_aiu_dst_names(result)
    assert order == ["A", "B"], f"Expected ['A', 'B'], got {order}"


def test_conditional_address_write_blocks_reordering():
    a = tir.decl_buffer((128,), "float16", name="A", scope="shared")
    b = tir.decl_buffer((128,), "float16", name="B", scope="shared")
    addr = tir.decl_buffer((1,), "int32", name="addr", scope="local")
    flag = tir.Var("flag", "bool")
    branch = tir.IfThenElse(
        flag,
        tir.SeqStmt(
            [
                _make_aiu_load(a),
                tir.BufferStore(addr, tir.IntImm("int32", 9), [0]),
            ]
        ),
        None,
    )
    body = tir.SeqStmt([branch, _make_aiu_load(b, tir.BufferLoad(addr, [0])), _make_use(b), _make_use(a)])
    result = _run_pass(body, [flag, a.data, b.data, addr.data], {a.data: a, b.data: b, addr.data: addr})
    tvm.ir.assert_structural_equal(result, body)


def test_loads_cannot_cross_a_conditional_side_effect():
    a = tir.decl_buffer((128,), "float16", name="A", scope="shared")
    b = tir.decl_buffer((128,), "float16", name="B", scope="shared")
    addr = tir.decl_buffer((1,), "int32", name="addr", scope="local")
    flag = tir.Var("flag", "bool")
    body = tir.SeqStmt(
        [
            _make_aiu_load(a),
            tir.IfThenElse(flag, tir.BufferStore(addr, tir.IntImm("int32", 9), [0]), None),
            _make_aiu_load(b, tir.BufferLoad(addr, [0])),
            _make_use(b),
            _make_use(a),
        ]
    )
    result = _run_pass(body, [flag, a.data, b.data, addr.data], {a.data: a, b.data: b, addr.data: addr})
    tvm.ir.assert_structural_equal(result, body)


if __name__ == "__main__":
    tilelang.testing.main()
