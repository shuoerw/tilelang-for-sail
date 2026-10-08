"""CPU-only checks of the fragment mapping required by bulk SWZL loads."""

import pytest

import tilelang
import tilelang.language as T
import tilelang.ppu.transform
from tilelang import tvm
from tilelang.layout import (
    make_full_bank_swizzled_layout,
    make_gemm_fragment_8x8_transposed,
    make_half_bank_swizzled_layout,
)


def _lower_copy(dtype, shared_shape, fragment_shape, layout, swizzle_fn, threads=32):
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})

    @T.prim_func
    def main():
        with T.Kernel(1, threads=threads):
            shared = T.alloc_shared(shared_shape, dtype)
            fragment = T.alloc_fragment(fragment_shape, dtype)
            T.annotate_layout({shared: swizzle_fn(shared), fragment: layout})
            T.copy(shared[: fragment_shape[0], : fragment_shape[1]], fragment)

    mod = tvm.IRModule.from_expr(main)
    with target, tvm.transform.PassContext(opt_level=3):
        mod = tvm.tirx.transform.BindTarget(target)(mod)
        mod = tilelang.transform.MaterializeKernelLaunch()(mod)
        mod = tilelang.ppu.transform.LayoutInference()(mod)
        mod = tilelang.ppu.transform.LowerTileOp()(mod)

    calls = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.Op):
            calls.append(node)

    tvm.tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    return calls


@pytest.mark.parametrize("dtype,transposed", [("bfloat16", False), ("bfloat16", True), ("int8", False), ("float4_e2m1fn", False)])
@pytest.mark.parametrize("swizzle_fn", [make_full_bank_swizzled_layout, make_half_bank_swizzled_layout])
@pytest.mark.parametrize(
    "repeats,row_major,expect_swzl",
    [
        ((2, 2), True, True),
        ((2, 2), False, True),
        ((1, 4), True, False),
        ((2, 4), True, False),
        ((2, 4), False, True),
        ((4, 2), True, True),
        ((4, 2), False, False),
        ((4, 4), True, False),
        ((4, 4), False, False),
    ],
)
def test_swzl_fragment_packet_shape(dtype, transposed, swizzle_fn, repeats, row_major, expect_swzl):
    bits = tvm.DataType(dtype).bits
    step = 32 // bits
    matrix_width = 128 // bits
    base = (
        make_gemm_fragment_8x8_transposed()
        if transposed
        else T.Fragment((8, matrix_width), forward_fn=lambda i, j: (4 * i + j // step, j % step))
    )
    layout = base.repeat(repeats, repeat_on_thread=False, lower_dim_first=row_major)
    fragment_shape = (8 * repeats[0], matrix_width * repeats[1])
    shared_shape = (32, (1024 if swizzle_fn == make_full_bank_swizzled_layout else 512) // bits)
    calls = _lower_copy(dtype, shared_shape, fragment_shape, layout, swizzle_fn)
    bulk = [call for call in calls if call.op.name == "tl.tix_ldmatrix_swzl"]
    if expect_swzl:
        assert bulk, "A compatible 2x2 packet should retain the bulk load"
        assert all(int(call.args[0]) == int(transposed) for call in bulk)
        assert all(int(call.args[5]) == int(row_major) for call in bulk)
    else:
        assert not bulk, "A strip of four matrices must not become a 2x2 bulk load"
        assert any(call.op.name == "tl.ptx_ldmatrix" for call in calls)


@pytest.mark.parametrize("replicated", [False, True])
def test_swzl_compatible_multiwarp_fragment(replicated):
    base = T.Fragment((16, 16), forward_fn=lambda i, j: (4 * (i % 8) + j % 8 // 2, 4 * (i // 8) + 2 * (j // 8) + j % 2))
    layout = base.replicate(4) if replicated else base.repeat((2, 2), repeat_on_thread=True)
    shape = (16, 16) if replicated else (32, 32)
    calls = _lower_copy("bfloat16", (32, 64), shape, layout, make_full_bank_swizzled_layout, threads=128)
    assert any(call.op.name == "tl.tix_ldmatrix_swzl" for call in calls)
