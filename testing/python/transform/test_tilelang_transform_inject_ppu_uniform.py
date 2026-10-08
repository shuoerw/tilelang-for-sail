"""Tests for InjectPpuUniform pass.

Verifies that warp-aligned thread predicates are rewritten to use
ppu_to_uniform_b32 for uniform branch generation.
"""

import pytest
import tilelang
import tvm
from tvm import tirx, IRModule
from tvm.tirx import (
    Var,
    IterVar,
    PrimFunc,
    IntImm,
    Cast,
    AttrStmt,
    Call,
    IfThenElse,
    Evaluate,
    FloorDiv,
    FloorMod,
)
from tvm.ir import Range


PPU_TARGET = tvm.target.Target({"kind": "ppu"})


def _make_func_with_guard(cond_fn, thread_extent=256):
    """Helper: create a PrimFunc with thread_extent AttrStmt and an if-guard.

    Args:
        cond_fn: callable(tx: Var) -> PrimExpr, builds the if-condition
        thread_extent: the thread extent for threadIdx.x
    Returns:
        PrimFunc with the structure:
            attr [iter_var] "thread_extent" = <extent>:
                if <cond_fn(tx)>:
                    evaluate(0)
    """
    tx = Var("threadIdx.x", dtype="int32")
    iv = IterVar(Range(0, thread_extent), tx, 0, "threadIdx.x")

    cond = cond_fn(tx)
    body = Evaluate(IntImm("int32", 0))
    if_stmt = IfThenElse(cond, body, None)

    attr_stmt = AttrStmt(iv, "thread_extent", IntImm("int32", thread_extent), if_stmt)
    return PrimFunc([], attr_stmt).with_attr("target", PPU_TARGET)


def _apply_pass(func):
    """Apply InjectPpuUniform pass to a PrimFunc."""
    mod = IRModule({"main": func})
    mod = tilelang.ppu.transform.InjectPpuUniform()(mod)
    return mod["main"]


def _has_ppu_to_uniform_b32(stmt):
    """Check if the TIR statement contains a ppu_to_uniform_b32 call."""
    found = [False]

    def visitor(node):
        if hasattr(node, "op") and hasattr(node.op, "name") and "ppu_to_uniform_b32" in str(node.op):
            found[0] = True

    tirx.stmt_functor.post_order_visit(stmt, visitor)
    return found[0]


def _get_if_condition(func):
    """Extract the condition from the IfThenElse inside the AttrStmt."""
    body = func.body
    # body is AttrStmt
    assert isinstance(body, tirx.AttrStmt), f"Expected AttrStmt, got {type(body)}"
    inner = body.body
    if isinstance(inner, tirx.IfThenElse):
        return inner.condition
    return None


def _make_func_with_expr(expr_fn, thread_extent=256):
    tx = Var("threadIdx.x", dtype="int32")
    expr = expr_fn(tx)
    iv = IterVar(Range(0, thread_extent), tx, 0, "threadIdx.x")
    attr_stmt = AttrStmt(iv, "thread_extent", IntImm("int32", thread_extent), Evaluate(expr))
    return PrimFunc([], attr_stmt).with_attr("target", PPU_TARGET)


def _count_ppu_uniform_calls(expr):
    found = [0]

    def visitor(node):
        if hasattr(node, "op") and "ppu_to_uniform_b32" in str(node.op):
            found[0] += 1

    tirx.stmt_functor.post_order_visit(expr, visitor)
    return found[0]


class TestInjectPpuUniform:
    """Test suite for the InjectPpuUniform pass."""

    def test_threadIdx_lt_128(self):
        """threadIdx.x < 128 → ppu_to_uniform_b32(tx>>5) < 4"""
        func = _make_func_with_guard(lambda tx: tirx.LT(tx, IntImm("int32", 128)), thread_extent=256)
        result = _apply_pass(func)
        cond = _get_if_condition(result)
        assert cond is not None, "If-guard should still exist"
        assert _has_ppu_to_uniform_b32(result.body), "Should contain ppu_to_uniform_b32"
        # Verify it's LT with rhs=4
        assert isinstance(cond, tirx.LT), f"Expected LT, got {type(cond)}"
        rhs = cond.b
        assert isinstance(rhs, tirx.IntImm) and rhs.value == 4, f"Expected rhs=4, got {rhs}"

    def test_threadIdx_ge_64(self):
        """threadIdx.x >= 64 → ppu_to_uniform_b32(tx>>5) >= 2"""
        func = _make_func_with_guard(lambda tx: tirx.GE(tx, IntImm("int32", 64)), thread_extent=256)
        result = _apply_pass(func)
        cond = _get_if_condition(result)
        assert cond is not None
        assert _has_ppu_to_uniform_b32(result.body)
        assert isinstance(cond, tirx.GE), f"Expected GE, got {type(cond)}"
        rhs = cond.b
        assert isinstance(rhs, tirx.IntImm) and rhs.value == 2, f"Expected rhs=2, got {rhs}"

    def test_threadIdx_rshift6_lt_4(self):
        """(threadIdx.x >> 6) < 4, i.e. FloorDiv(tx, 64) < 4
        Since shift=6 >= 5, result is ppu_to_uniform_b32(tx // 64) < 4"""
        func = _make_func_with_guard(
            lambda tx: tirx.LT(FloorDiv(tx, IntImm("int32", 64)), IntImm("int32", 4)),
            thread_extent=256,
        )
        result = _apply_pass(func)
        cond = _get_if_condition(result)
        assert cond is not None
        assert _has_ppu_to_uniform_b32(result.body)
        assert isinstance(cond, tirx.LT)
        rhs = cond.b
        assert isinstance(rhs, tirx.IntImm) and rhs.value == 4, f"Expected rhs=4, got {rhs}"

    def test_threadIdx_masked_lt_64(self):
        """(threadIdx.x % 128) < 64 → ppu_to_uniform_b32((tx%128)>>5) < 2
        base = tx % 128, shift=0, rhs=64
        effective_boundary = 64 * 1 = 64, 64 % 32 == 0 ✓
        additional_shift = 5-0 = 5, new_rhs = 64 / 32 = 2"""
        func = _make_func_with_guard(
            lambda tx: tirx.LT(FloorMod(tx, IntImm("int32", 128)), IntImm("int32", 64)),
            thread_extent=256,
        )
        result = _apply_pass(func)
        cond = _get_if_condition(result)
        assert cond is not None
        assert _has_ppu_to_uniform_b32(result.body)
        assert isinstance(cond, tirx.LT)
        rhs = cond.b
        assert isinstance(rhs, tirx.IntImm) and rhs.value == 2, f"Expected rhs=2, got {rhs}"

    def test_not_warp_aligned_unchanged(self):
        """threadIdx.x < 48 → unchanged (48 not multiple of 32)"""
        func = _make_func_with_guard(lambda tx: tirx.LT(tx, IntImm("int32", 48)), thread_extent=256)
        result = _apply_pass(func)
        cond = _get_if_condition(result)
        assert cond is not None
        # Should NOT contain ppu_to_uniform_b32 in the condition
        # The condition should remain as-is
        assert not _has_ppu_to_uniform_b32(IfThenElse(cond, Evaluate(IntImm("int32", 0)), None)), (
            "Should NOT rewrite non-warp-aligned predicate"
        )

    def test_full_extent_guard_unchanged(self):
        """threadIdx.x < 256 with 256 threads.
        Note: this guard would normally be eliminated by Simplify/CanProve
        before this pass runs. If it still exists, we still rewrite it
        since 256 % 32 == 0."""
        func = _make_func_with_guard(lambda tx: tirx.LT(tx, IntImm("int32", 256)), thread_extent=256)
        result = _apply_pass(func)
        cond = _get_if_condition(result)
        assert cond is not None
        # 256 % 32 == 0, so this should be rewritten
        assert _has_ppu_to_uniform_b32(result.body)

    def test_already_has_ppu_to_uniform_b32(self):
        """Already wrapped in ppu_to_uniform_b32 → unchanged"""
        from tvm.tirx import Call
        from tvm.tirx.op import Op

        # Get the ppu_to_uniform_b32 Op
        ppu_to_uniform_b32_op = Op.get("tl.ppu_to_uniform_b32")

        # Build: ppu_to_uniform_b32(tx // 32) < 4
        tx = Var("threadIdx.x", dtype="int32")
        iv = IterVar(Range(0, 256), tx, 0, "threadIdx.x")

        warp_id = FloorDiv(tx, IntImm("int32", 32))
        scalar_warp = Call("int32", ppu_to_uniform_b32_op, [warp_id])
        cond = tirx.LT(scalar_warp, IntImm("int32", 4))

        body = Evaluate(IntImm("int32", 0))
        if_stmt = IfThenElse(cond, body, None)
        attr_stmt = AttrStmt(iv, "thread_extent", IntImm("int32", 256), if_stmt)
        func = PrimFunc([], attr_stmt).with_attr("target", PPU_TARGET)

        result = _apply_pass(func)
        result_cond = _get_if_condition(result)
        # The condition should remain identical (not double-wrapped)
        assert result_cond is not None
        assert _count_ppu_uniform_calls(result.body) == 1
        tvm.ir.assert_structural_equal(result.body, func.body)
        # Check the RHS is still 4 (not further divided)
        assert isinstance(result_cond, tirx.LT)
        rhs = result_cond.b
        assert isinstance(rhs, tirx.IntImm) and rhs.value == 4

    def test_cast_threadIdx_lt_128(self):
        """Cast(int32, threadIdx.x) < 128 → ppu_to_uniform_b32(cast(tx)>>5) < 4"""
        func = _make_func_with_guard(
            lambda tx: tirx.LT(Cast("int32", tx), IntImm("int32", 128)),
            thread_extent=256,
        )
        result = _apply_pass(func)
        cond = _get_if_condition(result)
        assert cond is not None
        assert _has_ppu_to_uniform_b32(result.body)
        assert isinstance(cond, tirx.LT)
        rhs = cond.b
        assert isinstance(rhs, tirx.IntImm) and rhs.value == 4

    def test_right_shift_op_lt(self):
        """right_shift(tx, 6) < 4 → ppu_to_uniform_b32(tx>>6) < 4 (shift>=5)"""
        from tvm.tirx import Call
        from tvm.tirx.op import Op

        right_shift_op = Op.get("tirx.shift_right")
        func = _make_func_with_guard(
            lambda tx: tirx.LT(Call("int32", right_shift_op, [tx, IntImm("int32", 6)]), IntImm("int32", 4)), thread_extent=256
        )
        result = _apply_pass(func)
        cond = _get_if_condition(result)
        assert cond is not None
        assert _has_ppu_to_uniform_b32(result.body)
        assert isinstance(cond, tirx.LT)
        assert cond.b.value == 4

    def test_bitwise_and_lt_64(self):
        """bitwise_and(tx, 127) < 64 → ppu_to_uniform_b32((tx&127)/32) < 2"""
        from tvm.tirx import Call
        from tvm.tirx.op import Op

        bitwise_and_op = Op.get("tirx.bitwise_and")
        func = _make_func_with_guard(
            lambda tx: tirx.LT(Call("int32", bitwise_and_op, [tx, IntImm("int32", 127)]), IntImm("int32", 64)), thread_extent=256
        )
        result = _apply_pass(func)
        cond = _get_if_condition(result)
        assert cond is not None
        assert _has_ppu_to_uniform_b32(result.body)
        assert isinstance(cond, tirx.LT)
        assert cond.b.value == 2

    def test_bitwise_and_rshift_combo(self):
        """FloorDiv(bitwise_and(tx, 255), 64) < 2
        base = tx & 255, shift = 6 (from /64), shift>=5
        → ppu_to_uniform_b32((tx&255)/64) < 2"""
        from tvm.tirx import Call
        from tvm.tirx.op import Op

        bitwise_and_op = Op.get("tirx.bitwise_and")
        func = _make_func_with_guard(
            lambda tx: tirx.LT(
                FloorDiv(Call("int32", bitwise_and_op, [tx, IntImm("int32", 255)]), IntImm("int32", 64)), IntImm("int32", 2)
            ),
            thread_extent=256,
        )
        result = _apply_pass(func)
        cond = _get_if_condition(result)
        assert cond is not None
        assert _has_ppu_to_uniform_b32(result.body)
        assert isinstance(cond, tirx.LT)
        assert cond.b.value == 2

    def test_compound_and_both_rewritten(self):
        """And(tx < 128, tx >= 64) → both sub-items rewritten"""
        func = _make_func_with_guard(
            lambda tx: tirx.And(tirx.LT(tx, IntImm("int32", 128)), tirx.GE(tx, IntImm("int32", 64))), thread_extent=256
        )
        result = _apply_pass(func)
        cond = _get_if_condition(result)
        assert cond is not None
        assert _has_ppu_to_uniform_b32(result.body)
        # Both conjuncts should be rewritten (And of two ppu_to_uniform_b32 comparisons)
        assert isinstance(cond, tirx.And)

    def test_mod64_lt_16_not_aligned(self):
        """(tx % 64) < 16 → unchanged (16 * 1 = 16, 16 % 32 != 0)"""
        func = _make_func_with_guard(lambda tx: tirx.LT(FloorMod(tx, IntImm("int32", 64)), IntImm("int32", 16)), thread_extent=256)
        result = _apply_pass(func)
        cond = _get_if_condition(result)
        assert cond is not None
        assert not _has_ppu_to_uniform_b32(IfThenElse(cond, Evaluate(IntImm("int32", 0)), None)), "Should NOT rewrite (16 not warp-aligned)"

    def test_rshift3_lt_4_effective_32(self):
        """FloorDiv(tx, 8) < 4
        shift=3, rhs=4, effective_boundary=4*8=32, 32%32==0 ✓
        additional_shift=5-3=2, new_rhs=4/4=1
        → ppu_to_uniform_b32(tx/32) < 1"""
        func = _make_func_with_guard(lambda tx: tirx.LT(FloorDiv(tx, IntImm("int32", 8)), IntImm("int32", 4)), thread_extent=256)
        result = _apply_pass(func)
        cond = _get_if_condition(result)
        assert cond is not None
        assert _has_ppu_to_uniform_b32(result.body)
        assert isinstance(cond, tirx.LT)
        assert cond.b.value == 1, f"Expected rhs=1, got {cond.b.value}"

    def test_rshift5_ge_2(self):
        """FloorDiv(tx, 32) >= 2 → ppu_to_uniform_b32(tx/32) >= 2 (shift=5>=5)"""
        func = _make_func_with_guard(lambda tx: tirx.GE(FloorDiv(tx, IntImm("int32", 32)), IntImm("int32", 2)), thread_extent=256)
        result = _apply_pass(func)
        cond = _get_if_condition(result)
        assert cond is not None
        assert _has_ppu_to_uniform_b32(result.body)
        assert isinstance(cond, tirx.GE)
        assert cond.b.value == 2

    def test_rshift_op_ge_2(self):
        """right_shift(tx, 5) >= 2 → ppu_to_uniform_b32(tx>>5) >= 2"""
        from tvm.tirx import Call
        from tvm.tirx.op import Op

        right_shift_op = Op.get("tirx.shift_right")
        func = _make_func_with_guard(
            lambda tx: tirx.GE(Call("int32", right_shift_op, [tx, IntImm("int32", 5)]), IntImm("int32", 2)), thread_extent=256
        )
        result = _apply_pass(func)
        cond = _get_if_condition(result)
        assert cond is not None
        assert _has_ppu_to_uniform_b32(result.body)
        assert isinstance(cond, tirx.GE)
        assert cond.b.value == 2

    def test_lt_32_minimal(self):
        """tx < 32 → ppu_to_uniform_b32(tx>>5) < 1 (minimal warp-aligned)"""
        func = _make_func_with_guard(lambda tx: tirx.LT(tx, IntImm("int32", 32)), thread_extent=256)
        result = _apply_pass(func)
        cond = _get_if_condition(result)
        assert cond is not None
        assert _has_ppu_to_uniform_b32(result.body)
        assert isinstance(cond, tirx.LT)
        assert cond.b.value == 1

    def test_gt_not_matched(self):
        """tx > 64 → GT is not matched (only LT and GE are handled)"""
        func = _make_func_with_guard(lambda tx: tirx.GT(tx, IntImm("int32", 64)), thread_extent=256)
        result = _apply_pass(func)
        # GT is not LT or GE, should remain unchanged
        cond = _get_if_condition(result)
        assert cond is not None
        assert not _has_ppu_to_uniform_b32(IfThenElse(cond, Evaluate(IntImm("int32", 0)), None))

    def test_aiu_load_warp_operand_is_wrapped(self):
        """AIU arguments continue to use the original uniform wrapping."""
        op = tirx.op.Op.get("tl.ppu_aiu_load")
        args = [IntImm("int32", index) for index in range(10)]
        func = _make_func_with_expr(
            lambda tx: Call(
                "int32",
                op,
                [FloorDiv(tx, IntImm("int32", 32)) if index == 8 else arg for index, arg in enumerate(args)],
            )
        )

        result = _apply_pass(func)

        assert _count_ppu_uniform_calls(result.body) == 1

    def test_ldmatrix_swizzle_warp_operand_is_wrapped(self):
        op = tirx.op.Op.get("tl.tix_ldmatrix_swzl")
        args = [IntImm("int32", index) for index in range(6)]
        func = _make_func_with_expr(
            lambda tx: Call(
                "int32",
                op,
                [FloorDiv(tx, IntImm("int32", 32)) if index == 5 else arg for index, arg in enumerate(args)],
            )
        )

        result = _apply_pass(func)

        assert _count_ppu_uniform_calls(result.body) == 1

    def test_ldmatrix_swizzle_right_shift_operand_is_wrapped(self):
        op = tirx.op.Op.get("tl.tix_ldmatrix_swzl")
        right_shift_op = tirx.op.Op.get("tirx.shift_right")
        func = _make_func_with_expr(
            lambda tx: Call(
                "int32",
                op,
                [
                    Call(
                        "int32",
                        right_shift_op,
                        [tx, IntImm("int32", 5)],
                    )
                    if index == 5
                    else IntImm("int32", index)
                    for index in range(6)
                ],
            )
        )

        result = _apply_pass(func)

        assert _count_ppu_uniform_calls(result.body) == 1

    def test_ldmatrix_swizzle_masked_then_shifted_operand_is_wrapped(self):
        op = tirx.op.Op.get("tl.tix_ldmatrix_swzl")
        bitwise_and_op = tirx.op.Op.get("tirx.bitwise_and")
        func = _make_func_with_expr(
            lambda tx: Call(
                "int32",
                op,
                [
                    FloorDiv(
                        Call(
                            "int32",
                            bitwise_and_op,
                            [tx, IntImm("int32", 255)],
                        ),
                        IntImm("int32", 64),
                    )
                    if index == 5
                    else IntImm("int32", index)
                    for index in range(6)
                ],
            )
        )

        result = _apply_pass(func)

        assert _count_ppu_uniform_calls(result.body) == 1

    def test_ldmatrix_swizzle_non_warp_period_modulo_is_wrapped(self):
        """The swizzle offset uses t-heads-style direct thread replacement."""
        op = tirx.op.Op.get("tl.tix_ldmatrix_swzl")
        func = _make_func_with_expr(
            lambda tx: Call(
                "int32",
                op,
                [
                    FloorDiv(
                        FloorMod(tx, IntImm("int32", 33)),
                        IntImm("int32", 32),
                    )
                    if index == 5
                    else IntImm("int32", index)
                    for index in range(6)
                ],
            )
        )

        result = _apply_pass(func)

        assert _count_ppu_uniform_calls(result.body) == 1

    def test_ldmatrix_swizzle_non_offset_operand_is_wrapped(self):
        op = tirx.op.Op.get("tl.tix_ldmatrix_swzl")
        args = [IntImm("int32", index) for index in range(6)]
        func = _make_func_with_expr(
            lambda tx: Call(
                "int32",
                op,
                [FloorDiv(tx, IntImm("int32", 32)) if index == 4 else arg for index, arg in enumerate(args)],
            )
        )

        result = _apply_pass(func)

        assert _count_ppu_uniform_calls(result.body) == 1

    def test_raw_thread_operand_is_wrapped(self):
        op = tirx.op.Op.get("tl.ppu_aiu_load")
        args = [IntImm("int32", index) for index in range(10)]
        args[7] = IntImm("int32", 7)
        func = _make_func_with_expr(
            lambda tx: Call(
                "int32",
                op,
                [tx if index == 8 else arg for index, arg in enumerate(args)],
            )
        )

        result = _apply_pass(func)

        assert _count_ppu_uniform_calls(result.body) == 1

    def test_warp_uniform_expression_outside_supported_statements_is_not_wrapped(self):
        func = _make_func_with_expr(lambda tx: FloorDiv(tx, IntImm("int32", 32)) + IntImm("int32", 1))

        result = _apply_pass(func)

        assert _count_ppu_uniform_calls(result.body) == 0

    def test_thread_bound_relative_warp_expression_is_not_wrapped(self):
        func = _make_func_with_expr(lambda tx: FloorDiv(tx - IntImm("int32", 0), IntImm("int32", 32)))

        result = _apply_pass(func)

        assert _count_ppu_uniform_calls(result.body) == 0

    def test_masked_warp_expression_outside_ldmatrix_is_not_wrapped(self):
        func = _make_func_with_expr(lambda tx: FloorDiv(FloorMod(tx, IntImm("int32", 64)), IntImm("int32", 32)))

        result = _apply_pass(func)

        assert _count_ppu_uniform_calls(result.body) == 0

    def test_sub_warp_expression_is_not_wrapped(self):
        func = _make_func_with_expr(lambda tx: FloorDiv(tx, IntImm("int32", 16)))

        result = _apply_pass(func)

        assert _count_ppu_uniform_calls(result.body) == 0

    def test_warp_index_helpers_are_not_wrapped(self):
        """Warp helpers perform the uniform conversion inside their C++ body."""
        for name in (
            "tl.get_warp_idx_sync",
            "tl.get_warp_idx",
            "tl.get_warp_group_idx",
        ):
            op = tirx.op.Op.get(name)
            expr = Call("int32", op, []) + IntImm("int32", 1)
            func = _make_func_with_expr(lambda tx, expr=expr: expr)

            result = _apply_pass(func)

            assert _count_ppu_uniform_calls(result.body) == 0

    def test_pass_reapplies_uniform_wrapper(self):
        op = tirx.op.Op.get("tl.tix_ldmatrix_swzl")
        args = [IntImm("int32", index) for index in range(6)]
        func = _make_func_with_expr(
            lambda tx: Call(
                "int32",
                op,
                [FloorDiv(tx, IntImm("int32", 32)) if index == 5 else arg for index, arg in enumerate(args)],
            )
        )

        once = _apply_pass(func)
        twice = _apply_pass(once)

        assert _count_ppu_uniform_calls(twice.body) == 2

    def test_non_thread_loop_var_is_not_wrapped(self):
        op = tirx.op.Op.get("tl.ppu_aiu_load")
        loop_var = Var("i", dtype="int32")
        args = [IntImm("int32", index) for index in range(10)]
        args[7] = loop_var
        func = _make_func_with_expr(lambda tx: Call("int32", op, args))

        result = _apply_pass(func)

        assert _count_ppu_uniform_calls(result.body) == 0

    def test_non_ppu_target_is_unchanged(self):
        tx = Var("threadIdx.x", dtype="int32")
        iv = IterVar(Range(0, 256), tx, 0, "threadIdx.x")
        body = IfThenElse(tx < 64, Evaluate(IntImm("int32", 1)), None)
        attr_stmt = AttrStmt(iv, "thread_extent", IntImm("int32", 256), body)
        func = PrimFunc([], attr_stmt).with_attr("target", tvm.target.Target({"kind": "cuda", "arch": "sm_80"}))
        module = IRModule({"main": func})

        result = tilelang.ppu.transform.InjectPpuUniform()(module)["main"]

        tvm.ir.assert_structural_equal(result, func)
        ppu_func = func.with_attr("target", PPU_TARGET)
        assert _count_ppu_uniform_calls(_apply_pass(ppu_func).body) == 1


@pytest.mark.parametrize("shift", range(7))
@pytest.mark.parametrize("boundary", [64, 128])
@pytest.mark.parametrize("comparison", [tirx.LT, tirx.GE])
@pytest.mark.parametrize("base_kind", ["thread", "modulo", "mask", "aligned_offset"])
def test_uniform_rewrite_preserves_every_lane(shift, boundary, comparison, base_kind):
    def condition(tx):
        base = {
            "thread": tx,
            "modulo": FloorMod(tx, 256),
            "mask": tirx.bitwise_and(tx, 255),
            "aligned_offset": tx - 32,
        }[base_kind]
        return comparison(FloorDiv(base, 1 << shift), IntImm("int32", boundary >> shift))

    func = _make_func_with_guard(condition)
    result = _apply_pass(func)
    before = _get_if_condition(func)
    after = _get_if_condition(result)
    assert isinstance(after.a, Call)
    assert after.a.op.name == "tl.ppu_to_uniform_b32"

    def at(expr, lane):
        if isinstance(expr, Var):
            assert expr.same_as(func.body.node.var)
            return lane
        if isinstance(expr, IntImm):
            return int(expr.value)
        if isinstance(expr, Call):
            assert expr.op.name == "tirx.bitwise_and"
            return at(expr.args[0], lane) & at(expr.args[1], lane)
        operations = {
            tirx.Add: lambda a, b: a + b,
            tirx.Sub: lambda a, b: a - b,
            FloorDiv: lambda a, b: a // b,
            FloorMod: lambda a, b: a % b,
            tirx.LT: lambda a, b: a < b,
            tirx.GE: lambda a, b: a >= b,
        }
        return operations[type(expr)](at(expr.a, lane), at(expr.b, lane))

    for warp_start in range(0, 256, 32):
        scalar = at(after.a.args[0], warp_start)
        actual = scalar < int(after.b) if comparison is tirx.LT else scalar >= int(after.b)
        for lane in range(warp_start, warp_start + 32):
            assert at(after.a.args[0], lane) == scalar, "Uniform operand varies within a warp"
            assert bool(at(before, lane)) == actual, f"Changed predicate at thread {lane}"


@pytest.mark.parametrize("offset", [-31, -1, 1, 7, 31])
@pytest.mark.parametrize("shift", [0, 3, 5])
def test_unaligned_thread_offset_is_not_uniformized(offset, shift):
    func = _make_func_with_guard(lambda tx: FloorDiv(tx + offset, 1 << shift) < (64 >> shift))
    result = _apply_pass(func)
    tvm.ir.assert_structural_equal(_get_if_condition(func), _get_if_condition(result))
    assert not _has_ppu_to_uniform_b32(result.body)


@pytest.mark.parametrize("case", ["dynamic_mask", "same_name", "wide_cast", "narrow_cast", "trunc_div"])
def test_unproven_thread_expression_is_not_uniformized(case):
    other = Var("threadIdx.x" if case == "same_name" else "mask", "int32")

    def condition(tx):
        expressions = {
            "dynamic_mask": tirx.bitwise_and(tx, other) < 64,
            "same_name": other < 64,
            "wide_cast": Cast("int64", tx) + (1 << 32) < tirx.IntImm("int64", 1 << 33),
            "narrow_cast": Cast("int32", Cast("int8", tx)) < 64,
            "trunc_div": tirx.Div(tx - 32, IntImm("int32", 32)) < 2,
        }
        return expressions[case]

    func = _make_func_with_guard(condition)
    result = _apply_pass(func)
    tvm.ir.assert_structural_equal(_get_if_condition(func), _get_if_condition(result))
    assert not _has_ppu_to_uniform_b32(result.body)


def test_partial_x_warps_are_not_uniformized():
    func = _make_func_with_guard(lambda tx: tx < 32, thread_extent=48)
    result = _apply_pass(func)
    assert not _has_ppu_to_uniform_b32(result.body)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
