"""AIU lowering must match the complete shared-memory address mapping."""

import pytest

import tilelang
import tilelang.language as T
import tilelang.ppu.transform
from tilelang import tvm
from tilelang.layout import Layout, make_full_bank_swizzled_layout, make_half_bank_swizzled_layout


def _layout(buffer, bank_bytes, kind):
    if kind == "canonical":
        factory = make_full_bank_swizzled_layout if bank_bytes == 128 else make_half_bank_swizzled_layout
        return factory(buffer)
    height, width = map(int, buffer.shape)
    bits = tvm.DataType(buffer.dtype).bits
    vector = 128 // bits
    block_width = bank_bytes * 8 // bits
    shift = vector.bit_length() - 1
    mask = bank_bytes // 16 - 1

    def forward(i, j):
        if kind == "odd_offset":
            return i * width + j + 1
        if kind == "odd_stride":
            return i * (width + 1) + j
        if kind == "row_major":
            plain = i * width + j
            return T.bitwise_xor(plain, T.shift_left(T.bitwise_and(T.shift_right(plain, shift + 3), mask), shift))
        row_xor = i % 8 if bank_bytes == 128 else i % 8 // 2
        offset = j // block_width * height * block_width + i * block_width
        offset += T.bitwise_xor(j % block_width // vector, row_xor) * vector + j % vector
        return offset + (height * width if kind == "offset" else 0)

    return Layout(buffer.shape, forward)


def _lower(dtype, bank_bytes, kind, threads):
    height = 16
    width = 2 * bank_bytes * 8 // tvm.DataType(dtype).bits
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})

    @T.prim_func
    def main(A: T.Tensor((height, 2 * width), dtype)):
        with T.Kernel(1, threads=threads):
            shared = T.alloc_shared((height, width), dtype)
            T.annotate_layout({shared: _layout(shared, bank_bytes, kind)})
            T.copy(A[:, width : 2 * width], shared, prefer_instruction="aiu")

    mod = tvm.IRModule.from_expr(main)
    with target, tvm.transform.PassContext(opt_level=3, config={"tl.disable_aiu_lower": False}):
        mod = tvm.tirx.transform.BindTarget(target)(mod)
        mod = tilelang.transform.MaterializeKernelLaunch()(mod)
        mod = tilelang.ppu.transform.LayoutInference()(mod)
        mod = tilelang.ppu.transform.LowerTileOp()(mod)
    calls = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.Op) and node.op.name == "tl.ppu_aiu_load":
            calls.append(node)

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    return calls


@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float8_e4m3fn", "float4_e2m1fn"])
@pytest.mark.parametrize("bank_bytes", [64, 128])
@pytest.mark.parametrize("threads", [32, 128])
@pytest.mark.parametrize("kind", ["canonical", "flat", "row_major", "offset"])
def test_aiu_requires_canonical_address_mapping(dtype, bank_bytes, threads, kind):
    calls = _lower(dtype, bank_bytes, kind, threads)
    if kind in ("row_major", "offset"):
        assert not calls, "Matching XOR bits cannot justify a different physical address mapping"
        return
    assert len(calls) == (2 if threads == 32 else 1)

    def const(expr):
        return int(tvm.arith.Analyzer().simplify(expr))

    for index, call in enumerate(calls):
        assert len(call.args) == 10
        assert const(call.args[2]) == 4 * bank_bytes, "Global width is measured in bytes"
        assert const(call.args[3]) == 16, "Global height is measured in rows"
        assert const(call.args[4]) == bank_bytes, "Instruction width is measured in bytes"
        assert const(call.args[6]) == 4 * bank_bytes, "Global row stride is measured in bytes"
        assert const(call.args[9]) == (0 if bank_bytes == 128 else 1)
        variables = []
        tvm.tirx.stmt_functor.post_order_visit(
            call.args[7], lambda node, variables=variables: variables.append(node) if isinstance(node, tvm.tirx.Var) else None
        )
        at_thread_zero = tvm.tirx.stmt_functor.substitute(call.args[7], {var: 0 for var in variables})
        start = int(tvm.arith.Analyzer().simplify(at_thread_zero))
        assert start == (2 + index) * bank_bytes


@pytest.mark.parametrize("kind", ["odd_offset", "odd_stride"])
def test_fp4_non_byte_aligned_layout_falls_back(kind):
    assert not _lower("float4_e2m1fn", 128, kind, 128)
