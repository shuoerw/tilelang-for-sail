"""CPU-only checks of the fragment mapping required by bulk SWZL loads."""

import pytest

import tilelang
import tilelang.ppu.language as T
import tilelang.ppu.transform
from tilelang import tvm
from tilelang.layout import (
    make_full_bank_swizzled_layout,
    make_gemm_fragment_8x8_transposed,
    make_half_bank_swizzled_layout,
    make_linear_layout,
)
from tilelang.transform import PassConfigKey


def _lower_copy(dtype, shared_shape, fragment_shape, layout, swizzle_fn, threads=32, pass_configs=None, arch=15):
    target = tvm.target.Target({"kind": "ppu", "arch": f"ppu_{arch}"})
    copy_shape = tuple(dim for dim in fragment_shape if dim != 1)

    @T.prim_func
    def main():
        with T.Kernel(1, threads=threads):
            shared = T.alloc_shared(shared_shape, dtype)
            fragment = T.alloc_fragment(fragment_shape, dtype)
            T.annotate_layout({shared: swizzle_fn(shared), fragment: layout})
            T.copy(shared[: copy_shape[0], : copy_shape[1]], fragment)

    mod = tvm.IRModule.from_expr(main)
    with target, tvm.transform.PassContext(opt_level=3, config=pass_configs or {}):
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


@pytest.mark.parametrize("arch", [10, 15])
@pytest.mark.parametrize("disabled", [None, False, True])
@pytest.mark.parametrize("dtype,transposed", [("bfloat16", False), ("bfloat16", True), ("int8", False), ("float4_e2m1fn", False)])
@pytest.mark.parametrize("swizzle_fn", [make_full_bank_swizzled_layout, make_half_bank_swizzled_layout])
def test_swzl_disable_pass_config(arch, disabled, dtype, transposed, swizzle_fn):
    bits = tvm.DataType(dtype).bits
    step = 32 // bits
    matrix_width = 128 // bits
    base = (
        make_gemm_fragment_8x8_transposed()
        if transposed
        else T.Fragment((8, matrix_width), forward_fn=lambda i, j: (4 * i + j // step, j % step))
    )
    layout = base.repeat((2, 2), repeat_on_thread=False)
    shape = (16, matrix_width * 2)
    shared_shape = (32, (1024 if swizzle_fn == make_full_bank_swizzled_layout else 512) // bits)
    pass_configs = {} if disabled is None else {PassConfigKey.TL_DISABLE_LDMAT_SWZL.value: disabled}
    calls = _lower_copy(dtype, shared_shape, shape, layout, swizzle_fn, pass_configs=pass_configs, arch=arch)
    loads = [call for call in calls if call.op.name in ("tl.ptx_ldmatrix", "tl.tix_ldmatrix_swzl")]
    expected_op = "tl.ptx_ldmatrix" if disabled or arch != 15 else "tl.tix_ldmatrix_swzl"
    assert loads
    assert all(call.op.name == expected_op for call in loads)
    assert all(int(call.args[0]) == int(transposed) and int(call.args[1]) == 4 for call in loads)


@pytest.mark.parametrize("replicated", [False, True])
def test_swzl_compatible_multiwarp_fragment(replicated):
    base = T.Fragment((16, 16), forward_fn=lambda i, j: (4 * (i % 8) + j % 8 // 2, 4 * (i // 8) + 2 * (j // 8) + j % 2))
    layout = base.replicate(4) if replicated else base.repeat((2, 2), repeat_on_thread=True)
    shape = (16, 16) if replicated else (32, 32)
    calls = _lower_copy("bfloat16", (32, 64), shape, layout, make_full_bank_swizzled_layout, threads=128)
    assert any(call.op.name == "tl.tix_ldmatrix_swzl" for call in calls)


@pytest.mark.parametrize(
    "dtype,expect_ldmatrix",
    [
        ("float16", True),
        ("bfloat16", True),
        ("int16", True),
        ("uint16", True),
        ("int8", True),
        ("uint8", True),
        ("float8_e4m3fn", True),
        ("float8_e5m2", True),
        ("float4_e2m1fn", True),
        ("int4", False),
        ("uint4", False),
        ("bool", False),
        ("float16x2", False),
        ("float32", False),
    ],
)
def test_ldsm_dtype_gate(dtype, expect_ldmatrix):
    bits = tvm.DataType(dtype).bits
    step = 32 // bits
    matrix_width = 4 * step
    base = T.Fragment((8, matrix_width), forward_fn=lambda i, j: (4 * i + j // step, j % step))
    layout = base.repeat((2, 2), repeat_on_thread=False)
    shape = (16, matrix_width * 2)
    calls = _lower_copy(dtype, shape, shape, layout, make_linear_layout)
    matrix_loads = [call for call in calls if call.op.name in ("tl.ptx_ldmatrix", "tl.tix_ldmatrix_swzl", "tl.tix_ldmatrix_b8")]
    assert bool(matrix_loads) == expect_ldmatrix


@pytest.mark.parametrize("swizzle_fn", [make_linear_layout, make_full_bank_swizzled_layout])
@pytest.mark.parametrize("disabled", [False, True])
def test_ldsm_fragment_extra_unit_dimension_fallback(swizzle_fn, disabled):
    shape = (1, 16, 16)
    layout = T.Fragment(
        shape,
        forward_fn=lambda unit, i, j: (4 * (i % 8) + j % 8 // 2, 4 * (i // 8) + 2 * (j // 8) + j % 2),
    )
    calls = _lower_copy(
        "bfloat16",
        (16, 64),
        shape,
        layout,
        swizzle_fn,
        pass_configs={PassConfigKey.TL_DISABLE_LDMAT_SWZL.value: disabled},
    )
    assert not any(call.op.name in ("tl.ptx_ldmatrix", "tl.tix_ldmatrix_swzl", "tl.tix_ldmatrix_b8") for call in calls)
