import pytest
import tilelang
import tilelang.testing
from tilelang import tvm

tir = tvm.tirx


def _rewrite(func, pass_name):
    if pass_name == "StorageRewrite":
        transform = tilelang.transform.StorageRewrite()
    else:
        transform = tvm.get_global_func("tl.transform.PointerValueTypeRewrite")()
    return transform(tvm.IRModule({"main": func}))["main"]


@pytest.mark.parametrize("pass_name", ["StorageRewrite", "PointerValueTypeRewrite"])
@pytest.mark.parametrize("index_dtype", ["int32", "int64"])
@pytest.mark.parametrize("offset", [0, 7])
@pytest.mark.parametrize("stride", [0, 1, 16384, 32767, 32768, 49152, 65536, 1048576, 2**32, 2**32 + 32768])
def test_scalar_stride_preserves_access(pass_name, index_dtype, offset, stride):
    if index_dtype == "int32" and stride >= 2**31:
        pytest.skip("The coefficient requires a 64-bit index")
    index = tir.Var("index", index_dtype)
    src = tir.decl_buffer((tir.IntImm("int64", 8 * stride + offset + 1),), "int32", name="src")
    dst = tir.decl_buffer((1,), "int32", name="dst")
    address = index * tir.IntImm(index_dtype, stride) + tir.IntImm(index_dtype, offset)
    body = tir.BufferStore(dst, tir.BufferLoad(src, [address]), [0])
    func = tir.PrimFunc([src.data, dst.data, index], body, buffer_map={src.data: src, dst.data: dst})

    result = _rewrite(func, pass_name)

    # These are scalar accesses regardless of the distance between addresses.
    tvm.ir.assert_structural_equal(result, func)
    assert result.body.value.dtype == "int32"
    assert result.body.value.buffer.dtype == "int32"


@pytest.mark.parametrize(
    "coefficients,lanes,expected_lanes",
    [
        ((0,), 4, 4),
        ((4, 4), 4, 4),
        ((8,), 4, 4),
        ((65538,), 3, 3),
        ((2**32 + 2,), 3, 3),
        ((4, 1), 4, 1),
        ((3,), 3, 3),
        ((32768,), 3, 1),
        ((49152,), 5, 1),
        ((65536,), 3, 1),
        ((1048576,), 3, 1),
        ((2**32,), 3, 1),
        ((3, 65536), 3, 1),
    ],
)
@pytest.mark.parametrize("offset", [0, 1])
def test_mixed_scalar_vector_coefficients(coefficients, lanes, expected_lanes, offset):
    index = tir.Var("index", "int64")
    size = tir.IntImm("int64", lanes * (8 * max(coefficients) + offset + 8))
    src = tir.decl_buffer((size,), "int32", name="src")
    vector_store = tir.BufferStore(src, tir.Broadcast(tir.IntImm("int32", 7), lanes), [tir.Ramp(0, 1, lanes)])
    addresses = [index * coefficient + offset for coefficient in coefficients]
    reads = [tir.Evaluate(tir.BufferLoad(src, [address])) for address in addresses]
    func = tir.PrimFunc([src.data, index], tir.SeqStmt([vector_store, *reads]), buffer_map={src.data: src})

    result = _rewrite(func, "PointerValueTypeRewrite")

    expected_dtype = "int32" if expected_lanes == 1 else f"int32x{expected_lanes}"
    analyzer = tvm.arith.Analyzer()
    store = result.body.seq[0]
    assert store.buffer.dtype == expected_dtype
    assert store.value.dtype == f"int32x{lanes}"
    if expected_lanes == 1:
        tvm.ir.assert_structural_equal(store.indices[0], vector_store.indices[0])
    else:
        assert analyzer.can_prove_equal(store.indices[0], 0)
    for stmt, address in zip(result.body.seq[1:], addresses):
        assert stmt.value.dtype == "int32"
        if expected_lanes == 1:
            assert isinstance(stmt.value, tir.BufferLoad)
            assert analyzer.can_prove_equal(stmt.value.indices[0], address)
        else:
            assert isinstance(stmt.value, tir.Shuffle)
            load = stmt.value.vectors[0]
            assert load.dtype == expected_dtype
            scalar_address = load.indices[0] * expected_lanes + stmt.value.indices[0]
            assert analyzer.can_prove_equal(scalar_address, address)


def test_mixed_scalar_base_types_preserve_declared_type():
    index = tir.Var("index", "int64")
    src = tir.decl_buffer((8 * 65536,), "int32", name="src")
    float_view = tir.decl_buffer(src.shape, "float32", name="float_view", data=src.data)
    body = tir.SeqStmt(
        [
            tir.BufferStore(src, tir.Broadcast(tir.IntImm("int32", 7), 4), [tir.Ramp(0, 1, 4)]),
            tir.Evaluate(tir.BufferLoad(src, [index * 65536])),
            tir.Evaluate(tir.BufferLoad(float_view, [index * 65536 + 1])),
        ]
    )
    func = tir.PrimFunc([src.data, index], body, buffer_map={src.data: src})
    result = _rewrite(func, "PointerValueTypeRewrite")
    tvm.ir.assert_structural_equal(result, func)


if __name__ == "__main__":
    tilelang.testing.main()
