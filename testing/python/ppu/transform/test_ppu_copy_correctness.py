"""Physical-address regressions for PPU AIU copy lowering."""

import pytest

import tilelang
import tilelang.ppu.language as T
import tilelang.ppu.transform
from tilelang import Layout, tvm
from tilelang.layout import make_full_bank_swizzled_layout, make_half_bank_swizzled_layout


TARGET = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})


def make_region_kernel(dtype, bank_bytes, row_start, col_blocks, thread_start, threads, leading=False):
    block = bank_bytes * 8 // tvm.DataType(dtype).bits
    rows, cols = 8, 2 * block
    shape = (2, 32, 4 * block) if leading else (32, 4 * block)
    col_start = col_blocks * block
    factory = make_full_bank_swizzled_layout if bank_bytes == 128 else make_half_bank_swizzled_layout

    @T.prim_func
    def main(A: T.Tensor((rows, cols), dtype), B: T.Tensor((rows, cols), dtype)):
        with T.Kernel(1, threads=thread_start + threads):
            tx = T.get_thread_binding()
            shared = T.alloc_shared(shape, dtype)
            T.annotate_layout({shared: factory(shared)})
            if tx >= thread_start and tx < thread_start + threads:
                if leading:
                    T.copy(A, shared[1, row_start : row_start + rows, col_start : col_start + cols])
                    T.copy(shared[1, row_start : row_start + rows, col_start : col_start + cols], B)
                else:
                    T.copy(A, shared[row_start : row_start + rows, col_start : col_start + cols])
                    T.copy(shared[row_start : row_start + rows, col_start : col_start + cols], B)

    return main


def make_unaligned_region_kernel(row_start, col_start):
    @T.prim_func
    def main(A: T.Tensor((8, 64), "float16"), B: T.Tensor((8, 64), "float16")):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((16, 128), "float16")
            T.annotate_layout({shared: make_full_bank_swizzled_layout(shared)})
            T.copy(A, shared[row_start : row_start + 8, col_start : col_start + 64])
            T.copy(shared[row_start : row_start + 8, col_start : col_start + 64], B)

    return main


def make_fp4_scalar_copy_kernel(stride=1, offset=1, single_thread=False, thread_start=0):
    @T.prim_func
    def main(A: T.Tensor((64,), "float4_e2m1fn"), B: T.Tensor((64,), "float16")):
        with T.Kernel(1, threads=thread_start + 128):
            tx = T.get_thread_binding()
            shared = T.alloc_shared((64,), "float4_e2m1fn")
            T.annotate_layout({shared: Layout((64,), lambda i: stride * i + offset)})
            if single_thread:
                if tx == thread_start:
                    T.copy(A, shared)
                    T.copy(shared, B)
            else:
                if tx >= thread_start:
                    T.copy(A, shared)
                    T.copy(shared, B)

    return main


def make_odd_width_kernel(width):
    @T.prim_func
    def main(A: T.Tensor((16, 512), "float4_e2m1fn"), B: T.Tensor((16, width), "float16")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((16, 512), "float4_e2m1fn")
            T.annotate_layout({shared: make_full_bank_swizzled_layout(shared)})
            T.copy(A[:, :width], shared[:, :width])
            T.copy(shared[:, :width], B)

    return main


def _lower(program):
    with TARGET, tvm.transform.PassContext(opt_level=3, config={"tl.disable_aiu_lower": False}):
        mod = tvm.tirx.transform.BindTarget(TARGET)(tvm.IRModule.from_expr(program))
        mod = tilelang.transform.MaterializeKernelLaunch()(mod)
        mod = tilelang.ppu.transform.LayoutInference()(mod)
        return tilelang.ppu.transform.LowerTileOp()(mod)["main"]


def _leaves(stmt, guards=()):
    if isinstance(stmt, tvm.tirx.IfThenElse):
        yield from _leaves(stmt.then_case, (*guards, stmt.condition))
        if stmt.else_case is not None:
            yield from _leaves(stmt.else_case, (*guards, tvm.tirx.Not(stmt.condition)))
    elif hasattr(stmt, "seq"):
        for child in stmt.seq:
            yield from _leaves(child, guards)
    elif hasattr(stmt, "body"):
        yield from _leaves(stmt.body, guards)
    elif hasattr(stmt, "block"):
        yield from _leaves(stmt.block, guards)
    else:
        yield stmt, guards


def _call(stmt, name):
    return (
        isinstance(stmt, tvm.tirx.Evaluate)
        and isinstance(stmt.value, tvm.tirx.Call)
        and isinstance(stmt.value.op, tvm.ir.Op)
        and stmt.value.op.name == name
    )


def _integer(expr, thread, writer=None):
    variables = []
    tvm.tirx.stmt_functor.post_order_visit(expr, lambda node: variables.append(node) if isinstance(node, tvm.tirx.Var) else None)
    values = {"tx": thread, "fp4_writer": writer}
    assert all(var.name in values and values[var.name] is not None for var in variables), variables
    result = tvm.arith.Analyzer().simplify(tvm.tirx.stmt_functor.substitute(expr, {var: values[var.name] for var in variables}))
    return int(result)


def _offset(row, col, rows, block, vector):
    row_xor = row % 8 if block // vector == 8 else row % 8 // 2
    return col // block * rows * block + row * block + ((col % block // vector) ^ row_xor) * vector + col % vector


@pytest.mark.parametrize("dtype", ["float16", "float8_e4m3fn", "float4_e2m1fn"])
@pytest.mark.parametrize("bank_bytes", [64, 128])
@pytest.mark.parametrize("thread_start,threads", [(0, 32), (128, 128)])
@pytest.mark.parametrize("row_start,col_blocks,leading", [(0, 0, False), (8, 1, False), (16, 1, True)])
def test_aiu_region_addresses_match_every_destination_element(dtype, bank_bytes, thread_start, threads, row_start, col_blocks, leading):
    program = _lower(make_region_kernel(dtype, bank_bytes, row_start, col_blocks, thread_start, threads, leading))
    loads = [(stmt.value, guards) for stmt, guards in _leaves(program.body) if _call(stmt, "tl.ppu_aiu_load")]
    assert loads
    bits = tvm.DataType(dtype).bits
    block, vector = bank_bytes * 8 // bits, 128 // bits
    seen = set()
    for thread in range(thread_start, thread_start + threads, 32):
        for load, guards in loads:
            if not all(_integer(guard, thread) for guard in guards):
                continue
            base = _integer(load.args[0].args[2], thread)
            col = _integer(load.args[7], thread) * 8 // bits
            row = _integer(load.args[8], thread)
            height = _integer(load.args[5], thread)
            for i in range(height):
                for j in range(block):
                    point = (row + i, col + j)
                    assert point not in seen
                    seen.add(point)
                    expected = _offset(row_start + row + i, col_blocks * block + col + j, 32, block, vector)
                    if leading:
                        expected += 32 * 4 * block
                    actual = base + _offset(i, j, height, block, vector)
                    assert actual == expected, (thread, point, actual, expected)
    assert seen == {(i, j) for i in range(8) for j in range(2 * block)}


@pytest.mark.parametrize("row_start,col_start", [(1, 0), (0, 1), (8, 32)])
def test_unaligned_aiu_region_falls_back(row_start, col_start):
    program = _lower(make_unaligned_region_kernel(row_start, col_start))
    assert not any(_call(stmt, "tl.ppu_aiu_load") for stmt, _ in _leaves(program.body))


def test_unaligned_thread_range_falls_back():
    program = _lower(make_region_kernel("float16", 128, 0, 0, 1, 32))
    assert not any(_call(stmt, "tl.ppu_aiu_load") for stmt, _ in _leaves(program.body))


@pytest.mark.parametrize("width", [255, 257])
def test_odd_fp4_width_falls_back(width):
    program = _lower(make_odd_width_kernel(width))
    assert not any(_call(stmt, "tl.ppu_aiu_load") for stmt, _ in _leaves(program.body))


@pytest.mark.parametrize("thread_start", [0, 128])
@pytest.mark.parametrize("stride,offset,single_thread", [(1, 1, False), (2, 0, False), (2, 1, False), (1, 1, True)])
def test_fp4_scalar_copy_has_one_active_writer(thread_start, stride, offset, single_thread):
    program = _lower(make_fp4_scalar_copy_kernel(stride, offset, single_thread, thread_start))
    stores = [
        (stmt, guards)
        for stmt, guards in _leaves(program.body)
        if isinstance(stmt, tvm.tirx.BufferStore) and stmt.buffer.dtype == "float4_e2m1fn"
    ]
    assert stores
    for _, guards in stores:
        for writer in (thread_start, thread_start + 1, thread_start + 31):
            active = [tx for tx in range(thread_start, thread_start + 128) if all(_integer(guard, tx, writer) for guard in guards)]
            assert set(active) <= {thread_start if single_thread else writer}


def make_fp4_packed_copy_kernel():
    @T.prim_func
    def main(A: T.Tensor((16, 64), "float4_e2m1fn"), B: T.Tensor((16, 64), "float16")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((16, 64), "float4_e2m1fn")
            T.annotate_layout({shared: Layout((16, 64), lambda i, j: i * 64 + j)})
            T.copy(A, shared)
            T.copy(shared, B)

    return main


def test_fp4_fallback_preserves_cp_async():
    program = _lower(make_fp4_packed_copy_kernel())
    assert any(_call(stmt, "tl.ptx_cp_async") for stmt, _ in _leaves(program.body))
    assert not any(_call(stmt, "tl.ppu_aiu_load") for stmt, _ in _leaves(program.body))


def make_thread_dependent_fp4_kernel(dependent, thread_start):
    @T.prim_func
    def main(A: T.Tensor((96,), "float4_e2m1fn"), B: T.Tensor((32,), "float16")):
        with T.Kernel(1, threads=thread_start + 32):
            tx = T.get_thread_binding()
            shared = T.alloc_shared((96,), "float4_e2m1fn")
            T.annotate_layout({shared: Layout((96,), lambda i: 2 * i)})
            offset = 2 * (tx - thread_start)
            if tx >= thread_start and tx < thread_start + 32:
                if dependent == "source":
                    T.copy(A[offset : offset + 32], shared[:32])
                    for i in T.Parallel(32):
                        B[i] = T.Cast("float16", shared[i])
                else:
                    T.copy(A[:32], shared[offset : offset + 32])
                    for i in T.Parallel(32):
                        B[i] = T.Cast("float16", shared[3 * i])

    return main


@pytest.mark.parametrize("dependent", ["source", "destination"])
@pytest.mark.parametrize("thread_start", [0, 128])
def test_thread_dependent_fp4_copy_preserves_addresses(dependent, thread_start):
    program = _lower(make_thread_dependent_fp4_kernel(dependent, thread_start))
    for lane in range(32):
        expected = (2 * lane, 3 * lane) if dependent == "source" else (6 * lane, lane)
        assert _physical_stores(program, thread_start + lane) == [expected]


def _physical_stores(program, thread):
    analyzer = tvm.arith.Analyzer()

    def integer(expr, bindings):
        variables = []
        tvm.tirx.stmt_functor.post_order_visit(expr, lambda node: variables.append(node) if isinstance(node, tvm.tirx.Var) else None)
        substitutions = dict(bindings)
        for var in variables:
            if var.name in ("tx", "ty", "tz", "bx", "by", "bz"):
                substitutions[var] = thread if var.name == "tx" else 0
        return int(analyzer.simplify(tvm.tirx.stmt_functor.substitute(expr, substitutions)))

    def visit(stmt, bindings):
        if isinstance(stmt, tvm.tirx.Bind):
            bindings[stmt.var] = integer(stmt.value, bindings)
        elif isinstance(stmt, tvm.tirx.For):
            start, extent = integer(stmt.min, bindings), integer(stmt.extent, bindings)
            for i in range(start, start + extent):
                yield from visit(stmt.body, {**bindings, stmt.loop_var: i})
        elif isinstance(stmt, tvm.tirx.IfThenElse):
            if integer(stmt.condition, bindings):
                yield from visit(stmt.then_case, bindings)
            elif stmt.else_case is not None:
                yield from visit(stmt.else_case, bindings)
        elif isinstance(stmt, tvm.tirx.BufferStore):
            loads = []
            tvm.tirx.stmt_functor.post_order_visit(
                stmt.value, lambda node: loads.append(node) if isinstance(node, tvm.tirx.BufferLoad) else None
            )
            if stmt.buffer.scope().startswith("shared"):
                assert len(loads) == 1
                yield integer(stmt.indices[0], bindings), integer(loads[0].indices[0], bindings)
        elif hasattr(stmt, "seq"):
            for child in stmt.seq:
                yield from visit(child, bindings)
        elif hasattr(stmt, "body"):
            yield from visit(stmt.body, bindings)
        elif hasattr(stmt, "block"):
            yield from visit(stmt.block, bindings)

    return list(visit(program.body, {}))


def make_fp4_fallback_kernel(dtype, destination, thread_start=0, threads=128):
    @T.prim_func
    def main(A: T.Tensor((64,), dtype), Packed: T.Tensor((66,), "float4_e2m1fn"), B: T.Tensor((64,), "float16")):
        with T.Kernel(1, threads=thread_start + threads):
            tx = T.get_thread_binding()
            shared = T.alloc_shared((64,), "float4_e2m1fn")
            T.annotate_layout({shared: Layout((64,), lambda i: i + 1)})
            if tx >= thread_start and tx < thread_start + threads:
                if destination == "global":
                    T.copy(A, Packed[1:65])
                    T.copy(Packed[1:65], B)
                else:
                    T.copy(A, shared)
                    T.copy(shared, B)

    return main
