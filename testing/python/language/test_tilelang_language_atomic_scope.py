"""Scoped atomic IR and target validation; no accelerator required."""

import pytest

import tilelang
import tilelang.language as T
from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.pass_pipeline.pipeline import PassPipeline
from tvm import IRModule, tirx
from tvm.error import InternalError
from tvm.ir import Op
from tvm.target import Target


_SCOPED_OPERATIONS = [(op, scope) for op in ("add", "add_ret", "load", "store") for scope in ("block", "device", "system")]


def _atomic_call(operation, buffer, index=0, scope="device"):
    if operation == "load":
        return T.atomic_load(buffer[index], memory_order="acquire", scope=scope)
    if operation == "store":
        return T.atomic_store(buffer[index], 1, memory_order="release", scope=scope)
    return T.atomic_add(buffer[index], 1, memory_order="release", scope=scope, return_prev=operation == "add_ret")


def _module(buffer, body):
    return IRModule({"main": tirx.PrimFunc([buffer.data], body, buffer_map={buffer.data: buffer})})


@pytest.mark.parametrize(("operation", "scope"), _SCOPED_OPERATIONS)
def test_atomic_scope_uses_shared_op(operation, scope):
    buffer = tirx.decl_buffer((4,), "int32")
    call = _atomic_call(operation, buffer, scope=scope)
    assert call.op.name == f"tl.atomic_{operation}_elem_op"
    legacy_arity = 2 if operation == "load" else 3
    assert len(call.args) == legacy_arity + (scope != "device")
    if scope != "device":
        assert call.args[-1].value == scope


_NATIVE_BACKENDS = {
    "ppu": ("target.build.tilelang_ppu_without_compile", {"kind": "ppu", "arch": "ppu_15"}),
    "cuda": ("target.build.tilelang_cuda_without_compile", {"kind": "cuda", "arch": "sm_80"}),
    "hip": ("target.build.tilelang_hip_without_compile", {"kind": "hip", "mcpu": "gfx90a"}),
    "cutedsl": ("target.build.tilelang_cutedsl_without_compile", {"kind": "cuda", "arch": "sm_80"}),
    "c": ("target.build.tilelang_c", "c"),
    "metal": ("target.build.tilelang_metal", "metal"),
    "llvm": ("target.build.llvm", "llvm"),
}


def _native_codegen(backend, buffer, call):
    builder_name, target_config = _NATIVE_BACKENDS[backend]
    builder = tilelang.tvm.ffi.get_global_func(builder_name, allow_missing=True)
    if builder is None:
        pytest.skip(f"{backend} codegen is not enabled in this build")
    target = Target(target_config)
    # Native codegen receives a lowered pointer rather than tl.access_ptr.
    if call.args:
        call = tirx.Call(call.dtype, call.op, [tirx.address_of(buffer[0]), *call.args[1:]])
    params = [buffer.data, *[arg for arg in call.args[1:] if isinstance(arg, tirx.Var)]]
    func = tirx.PrimFunc(params, tirx.Evaluate(call), buffer_map={buffer.data: buffer}).with_attr(
        {"global_symbol": "main", "calling_conv": 2, "target": target, "tirx.kernel_launch_params": []}
    )
    with target:
        return builder(IRModule({"main": func}), target)


@pytest.mark.parametrize("scope", ["block", "system"])
@pytest.mark.parametrize("entry", ["pipeline", "codegen", "codegen_without_compile"])
def test_backend_wrappers_delegate_scope_validation(scope, entry):
    buffer = tirx.decl_buffer((4,), "int32")
    mod = _module(buffer, tirx.Evaluate(_atomic_call("add", buffer, scope=scope)))
    target = Target("c")
    visited = []

    def lower(mod, target):
        visited.append((mod, target))
        return mod

    if entry == "pipeline":
        result = PassPipeline("test", lower).lower(mod, target)
    else:
        result = DeviceCodegen("test", lower, lower).lower(mod, target, compile_device=entry == "codegen")
    assert result is mod
    assert len(visited) == 1
    assert visited[0][0] is mod
    assert visited[0][1] is target


@pytest.mark.parametrize("operation", ["add", "add_ret", "load", "store"])
@pytest.mark.parametrize("scope", ["block", "system"])
@pytest.mark.parametrize("backend", ["cuda", "hip", "cutedsl", "c", "metal", "llvm"])
def test_native_codegen_rejects_unsupported_atomic_scope(operation, scope, backend):
    buffer = tirx.decl_buffer((4,), "int32", name="data")
    call = _atomic_call(operation, buffer, scope=scope)
    # LLVM has no lowering for these TileLang intrinsics at any scope.
    message = "tl.atomic_" if backend == "llvm" else "Non-device-scope atomics are not implemented"
    with pytest.raises(InternalError, match=message):
        _native_codegen(backend, buffer, call)


@pytest.mark.parametrize("operation", ["add", "add_ret", "load", "store"])
@pytest.mark.parametrize("explicit", [False, True])
@pytest.mark.parametrize("backend", ["ppu", "cuda", "hip", "cutedsl"])
def test_native_codegen_device_scope(operation, explicit, backend):
    buffer = tirx.decl_buffer((4,), "int32", name="data")
    call = _atomic_call(operation, buffer)
    if explicit:
        call = tirx.Call(call.dtype, call.op, [*call.args, tirx.StringImm("device")])
    source = _native_codegen(backend, buffer, call).inspect_source()
    assert f"Atomic{'AddRet' if operation == 'add_ret' else operation.capitalize()}(" in source


@pytest.mark.parametrize("return_prev", [False, True])
def test_native_codegen_legacy_atomic_add_without_order(return_prev):
    buffer = tirx.decl_buffer((4,), "int32", name="data")
    call = T.atomic_add(buffer[0], 1, return_prev=return_prev)
    assert len(call.args) == 2
    source = _native_codegen("ppu", buffer, call).inspect_source()
    assert ("AtomicAddRet(" if return_prev else "AtomicAdd(") in source


def _validate_with_vectorization(buffer, call):
    index = tirx.Var("i", "int32")
    loop = tirx.For(index, 0, 4, tirx.ForKind.VECTORIZED, tirx.Evaluate(call))
    tilelang.transform.VectorizeLoop()(_module(buffer, loop))


@pytest.mark.parametrize("operation", ["add", "add_ret", "load", "store"])
@pytest.mark.parametrize("scope", ["rank", 1, "dynamic"])
@pytest.mark.parametrize("entry", ["vectorize", "ppu", "cuda", "hip", "cutedsl", "c", "metal"])
def test_raw_atomic_scope_validation(operation, scope, entry):
    buffer = tirx.decl_buffer((4,), "int32")
    call = _atomic_call(operation, buffer)
    value = tirx.Var("scope", "int32") if scope == "dynamic" else scope
    raw = T.call_intrin(call.dtype, call.op, *call.args, value)
    with pytest.raises(InternalError, match="scope must be a constant string"):
        if entry == "vectorize":
            _validate_with_vectorization(buffer, raw)
        else:
            _native_codegen(entry, buffer, raw)


@pytest.mark.parametrize("operation", ["add", "add_ret", "load", "store"])
@pytest.mark.parametrize("case", ["empty", "too_few", "too_many"])
@pytest.mark.parametrize("entry", ["vectorize", "ppu", "cuda", "hip", "cutedsl", "c", "metal"])
def test_raw_atomic_arity_validation(operation, case, entry):
    buffer = tirx.decl_buffer((4,), "int32")
    call = _atomic_call(operation, buffer, scope="system")
    min_args = 3 if operation == "store" else 2
    args = [] if case == "empty" else [*call.args, 0] if case == "too_many" else list(call.args[: min_args - 1])
    raw = tirx.Call(call.dtype, call.op, args)
    with pytest.raises(InternalError, match="expects .* arguments"):
        if entry == "vectorize":
            _validate_with_vectorization(buffer, raw)
        else:
            _native_codegen(entry, buffer, raw)


@pytest.mark.parametrize("operation", ["add", "add_ret", "load", "store"])
def test_device_scope_preserves_unknown_order_error(operation):
    buffer = tirx.decl_buffer((4,), "int32")
    atomic = getattr(T, f"atomic_{'add' if operation == 'add_ret' else operation}")
    args = [buffer[0]] if operation == "load" else [buffer[0], 1]
    # Upstream validates load/store orders, but atomic_add still indexes the raw order map.
    error, match = (KeyError, None) if operation in ("add", "add_ret") else (ValueError, f"^atomic_{operation} does not support")
    with pytest.raises(error, match=match):
        atomic(*args, memory_order="unknown")


@pytest.mark.parametrize(("operation", "scope"), [(op, scope) for op, scope in _SCOPED_OPERATIONS if scope != "device"])
def test_vectorization_preserves_atomic_scope(operation, scope):
    buffer = tirx.decl_buffer((4,), "int32")
    index = tirx.Var("i", "int32")
    call = _atomic_call(operation, buffer, index, scope=scope)
    loop = tirx.For(index, 0, 4, tirx.ForKind.VECTORIZED, tirx.Evaluate(call))
    mod = tilelang.transform.VectorizeLoop()(_module(buffer, loop))
    calls, loops = [], []

    def visit(node):
        if isinstance(node, tirx.For):
            loops.append(node)
        if isinstance(node, tirx.Call) and isinstance(node.op, Op) and node.op.name == call.op.name:
            calls.append(node)

    tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    assert len(loops) == len(calls) == 1
    assert loops[0].kind == tirx.ForKind.SERIAL
    assert int(loops[0].extent) == 4
    assert calls[0].args[-1].value == scope
    assert int(calls[0].args[-2]) == (2 if operation == "load" else 3)
    pointer_index = calls[0].args[0].args[0].indices[0]
    assert pointer_index.same_as(loops[0].loop_var)


@pytest.mark.parametrize("return_prev", [False, True])
@pytest.mark.parametrize("scope", ["block", "system"])
@pytest.mark.parametrize("dtype", ["int32", "uint32", "int64", "uint64", "float32", "float64"])
def test_scoped_atomic_add_defaults_and_return_value(return_prev, scope, dtype):
    buffer = tirx.decl_buffer((4,), dtype)
    call = T.atomic_add(buffer[0], 1, scope=scope, return_prev=return_prev)
    assert call.op.name == ("tl.atomic_add_ret_elem_op" if return_prev else "tl.atomic_add_elem_op")
    assert str(call.dtype) == (dtype if return_prev else "handle")
    assert int(call.args[2]) == 0
    assert call.args[3].value == scope
    source = _native_codegen("ppu", buffer, call).inspect_source()
    assert ("PPUBlockAtomicAdd(" if scope == "block" else "PPUSystemAtomicAdd(") in source


@pytest.mark.parametrize("scope", ["block", "system"])
def test_scoped_atomic_add_rejects_unsupported_arguments(scope):
    buffer = tirx.decl_buffer((4,), "int32")
    with pytest.raises(NotImplementedError, match="only for scalar"):
        T.atomic_add(buffer, 1, scope=scope)
    with pytest.raises(NotImplementedError, match="use_tma"):
        T.atomic_add(buffer[0], 1, scope=scope, use_tma=True)
    with pytest.raises(ValueError, match="memory_order"):
        T.atomic_add(buffer[0], 1, scope=scope, memory_order="invalid")


@pytest.mark.parametrize("dtype", ["int8", "int16", "float16", "bfloat16", "bool"])
def test_block_atomic_add_rejects_subword_types(dtype):
    buffer = tirx.decl_buffer((4,), dtype)
    with pytest.raises(TypeError, match="32/64-bit"):
        T.atomic_add(buffer[0], 1, scope="block")


@pytest.mark.parametrize("scope", ["block", "system"])
def test_scoped_atomic_add_return_value_vectorization(scope):
    buffer = tirx.decl_buffer((4,), "int32")
    index = tirx.Var("i", "int32")
    call = T.atomic_add(buffer[index], 1, memory_order="acq_rel", scope=scope, return_prev=True)
    loop = tirx.For(index, 0, 4, tirx.ForKind.VECTORIZED, tirx.Evaluate(call))
    mod = tilelang.transform.VectorizeLoop()(_module(buffer, loop))
    loops, calls = [], []

    def visit(node):
        if isinstance(node, tirx.For):
            loops.append(node)
        if isinstance(node, tirx.Call) and node.op == call.op:
            calls.append(node)

    tirx.stmt_functor.post_order_visit(mod["main"].body, visit)
    assert len(loops) == len(calls) == 1
    assert loops[0].kind == tirx.ForKind.SERIAL
    assert int(loops[0].extent) == 4
    assert calls[0].args[-1].value == scope
    assert int(calls[0].args[-2]) == 4
    assert calls[0].args[0].args[0].indices[0].same_as(loops[0].loop_var)


@pytest.mark.parametrize("operation", ["add", "add_ret", "load", "store"])
@pytest.mark.parametrize("dtype", ["int8", "uint8", "int16", "uint16", "float16", "bfloat16", "bool"])
def test_system_atomic_rejects_subword_types(operation, dtype):
    buffer = tirx.decl_buffer((4,), dtype)
    with pytest.raises(TypeError, match="does not support dtype"):
        _atomic_call(operation, buffer, scope="system")


@pytest.mark.parametrize("operation", ["add", "add_ret", "load", "store"])
@pytest.mark.parametrize("dtype", ["int32", "uint32", "int64", "uint64", "float32", "float64"])
def test_system_atomic_accepts_supported_types(operation, dtype):
    buffer = tirx.decl_buffer((4,), dtype)
    call = _atomic_call(operation, buffer, scope="system")
    source = _native_codegen("ppu", buffer, call).inspect_source()
    assert "PPUSystemAtomic" in source
    if operation in ("add_ret", "load"):
        assert str(call.dtype) == dtype


@pytest.mark.parametrize("scope", ["block", "system"])
@pytest.mark.parametrize("operation", ["add", "add_ret", "load", "store"])
@pytest.mark.parametrize("memory_order", ["relaxed", "consume", "acquire", "release", "acq_rel", "seq_cst", "invalid", 6])
def test_scoped_atomic_operation_orders(scope, operation, memory_order):
    valid_orders = {
        "add": {"relaxed", "consume", "acquire", "release", "acq_rel", "seq_cst"},
        "add_ret": {"relaxed", "consume", "acquire", "release", "acq_rel", "seq_cst"},
        "load": {"relaxed", "consume", "acquire", "seq_cst"},
        "store": {"relaxed", "release", "seq_cst"},
    }
    buffer = tirx.decl_buffer((1,), "int32")
    atomic = getattr(T, f"atomic_{'add' if operation == 'add_ret' else operation}")
    args = [buffer[0]] if operation == "load" else [buffer[0], 1]
    kwargs = {"return_prev": operation == "add_ret"} if operation in ("add", "add_ret") else {}
    if memory_order in valid_orders[operation]:
        call = atomic(*args, memory_order=memory_order, scope=scope, **kwargs)
        expected_id = ["relaxed", "consume", "acquire", "release", "acq_rel", "seq_cst"].index(memory_order)
        assert int(call.args[-2]) == expected_id
        assert call.args[-1].value == scope
    else:
        with pytest.raises(ValueError, match=f"{scope}-scope atomic_.* does not support memory_order"):
            atomic(*args, memory_order=memory_order, scope=scope, **kwargs)


@pytest.mark.parametrize("operation", ["load", "store"])
@pytest.mark.parametrize("dtype", ["int8", "int16", "float16"])
def test_block_atomic_load_store_preserves_type_acceptance(operation, dtype):
    buffer = tirx.decl_buffer((1,), dtype)
    call = _atomic_call(operation, buffer, scope="block")
    assert call.args[-1].value == "block"
    if operation == "load":
        assert str(call.dtype) == dtype
