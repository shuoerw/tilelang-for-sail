"""Scoped atomic IR and target validation; no accelerator required."""

import pytest

import tilelang
import tilelang.language as T
from tilelang.backend.atomic import validate_atomic_scopes
from tilelang.backend.device_codegen import DeviceCodegen
from tilelang.backend.pass_pipeline.pipeline import PassPipeline
from tvm import IRModule, tirx
from tvm.error import InternalError
from tvm.ir import Op
from tvm.target import Target


_SCOPED_OPERATIONS = [(op, scope) for op in ("add", "load", "store") for scope in ("device", "system")] + [
    ("load", "block"),
    ("store", "block"),
]


def _atomic_call(operation, buffer, index=0, scope="device"):
    if operation == "load":
        return T.atomic_load(buffer[index], memory_order="acquire", scope=scope)
    if operation == "store":
        return T.atomic_store(buffer[index], 1, memory_order="release", scope=scope)
    return T.atomic_add(buffer[index], 1, memory_order="release", scope=scope)


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


@pytest.mark.parametrize("operation", ["add", "load", "store"])
@pytest.mark.parametrize("target", ["cuda", "hip", "metal", "c", "llvm"])
@pytest.mark.parametrize("entry", ["pipeline", "codegen", "codegen_without_compile"])
def test_system_scope_rejected_before_backend(operation, target, entry):
    buffer = tirx.decl_buffer((4,), "int32")
    mod = _module(buffer, tirx.Evaluate(_atomic_call(operation, buffer, scope="system")))
    target = Target({"kind": "cuda", "arch": "sm_80"} if target == "cuda" else target)

    def unreachable(mod, target):
        pytest.fail("Unsupported system scope reached the backend")

    with pytest.raises(ValueError, match="system scope is not implemented for target"):
        if entry == "pipeline":
            PassPipeline("test", unreachable).lower(mod, target)
        else:
            DeviceCodegen("test", unreachable, unreachable).lower(
                mod,
                target,
                compile_device=entry == "codegen",
            )


@pytest.mark.parametrize(("operation", "scope"), _SCOPED_OPERATIONS)
@pytest.mark.parametrize(
    ("backend", "target_config"),
    [
        ("cuda", {"kind": "cuda", "arch": "sm_80"}),
        ("hip", {"kind": "hip", "mcpu": "gfx90a"}),
        ("cutedsl", {"kind": "cuda", "arch": "sm_80"}),
    ],
)
def test_native_codegen_atomic_scope(operation, scope, backend, target_config):
    builder = tilelang.tvm.ffi.get_global_func(
        f"target.build.tilelang_{backend}_without_compile",
        allow_missing=True,
    )
    if builder is None:
        pytest.skip(f"{backend} codegen is not enabled in this build")
    target = Target(target_config)
    buffer = tirx.decl_buffer((4,), "int32", name="data")
    call = _atomic_call(operation, buffer, scope=scope)
    body = tirx.Evaluate(tirx.Call(call.dtype, call.op, [buffer.data, *call.args[1:]]))
    func = tirx.PrimFunc([buffer.data], body).with_attr(
        {
            "global_symbol": "main",
            "calling_conv": 2,
            "target": target,
        }
    )
    mod = IRModule({"main": func})
    # Invoke native codegen directly, without the Python validation wrappers.
    with target:
        if scope != "device":
            with pytest.raises(InternalError, match="Non-device-scope atomics are not implemented"):
                builder(mod, target)
        else:
            source = builder(mod, target).inspect_source()
            assert f"Atomic{operation.capitalize()}(" in source


@pytest.mark.parametrize("operation", ["add", "load", "store"])
@pytest.mark.parametrize("explicit", [False, True])
def test_device_scope_remains_valid(operation, explicit):
    buffer = tirx.decl_buffer((4,), "int32")
    call = _atomic_call(operation, buffer)
    if explicit:
        call = tirx.Call(call.dtype, call.op, [*call.args, tirx.StringImm("device")])
    mod = _module(buffer, tirx.Evaluate(call))
    assert PassPipeline("test", lambda mod, target: mod).lower(mod, Target("c")) is mod


@pytest.mark.parametrize("operation", ["add", "load", "store"])
@pytest.mark.parametrize("scope", ["rank", 1])
def test_raw_atomic_scope_validation(operation, scope):
    buffer = tirx.decl_buffer((4,), "int32")
    call = _atomic_call(operation, buffer)
    raw = T.call_intrin(call.dtype, call.op, *call.args, scope)
    with pytest.raises(ValueError, match="scope must be a constant string"):
        validate_atomic_scopes(_module(buffer, tirx.Evaluate(raw)), Target("c"))


@pytest.mark.parametrize("operation", ["add", "load", "store"])
@pytest.mark.parametrize("extra", [False, True])
def test_raw_atomic_arity_validation(operation, extra):
    buffer = tirx.decl_buffer((4,), "int32")
    call = _atomic_call(operation, buffer, scope="system")
    args = [*call.args, 0] if extra else [call.args[0]]
    raw = tirx.Call(call.dtype, call.op, args)
    with pytest.raises(ValueError, match="expects .* arguments"):
        validate_atomic_scopes(_module(buffer, tirx.Evaluate(raw)), Target("c"))


@pytest.mark.parametrize("operation", ["add", "load", "store"])
def test_device_scope_preserves_unknown_order_error(operation):
    buffer = tirx.decl_buffer((4,), "int32")
    atomic = getattr(T, f"atomic_{operation}")
    args = [buffer[0]] if operation == "load" else [buffer[0], 1]
    with pytest.raises(KeyError):
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


def test_block_scope_validation():
    buffer = tirx.decl_buffer((4,), "int32")
    for operation in ("load", "store"):
        mod = _module(buffer, tirx.Evaluate(_atomic_call(operation, buffer, scope="block")))
        validate_atomic_scopes(mod, Target({"kind": "ppu", "arch": "ppu_15"}))
        with pytest.raises(ValueError, match="block scope is not implemented"):
            validate_atomic_scopes(mod, Target("c"))
    with pytest.raises(ValueError, match="Unsupported memory scope"):
        _atomic_call("add", buffer, scope="block")
    call = _atomic_call("add", buffer)
    raw = tirx.Call(call.dtype, call.op, [*call.args, tirx.StringImm("block")])
    with pytest.raises(ValueError, match="block scope is supported only"):
        validate_atomic_scopes(_module(buffer, tirx.Evaluate(raw)), Target({"kind": "ppu", "arch": "ppu_15"}))
