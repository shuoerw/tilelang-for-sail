import inspect

import tilelang.language.common as common
import tilelang.ppu.language as T
from tvm import tirx
from tvm.tirx.stmt_functor import post_order_visit


def _keywords(func):
    return set(inspect.signature(func).parameters)


def test_ppu_dialect_declares_target_hints():
    expected = {
        "Parallel": {"prefer_async"},
        "reduce_max": {"nan_propagate"},
        "reduce_min": {"nan_propagate"},
        "reduce_absmax": {"nan_propagate"},
        "unroll": {"unroll_factor"},
        "Unroll": {"unroll_factor"},
    }
    for name, extra in expected.items():
        assert _keywords(getattr(T, name)) - _keywords(getattr(common, name)) == extra
        assert getattr(T, name) is not getattr(common, name)
        assert name in T.__all__


def test_ppu_loop_hints_are_recorded():
    @T.prim_func
    def loops(A: T.Tensor((64,), "float32"), B: T.Tensor((64,), "float32")):
        with T.Kernel(1, threads=64):
            for i in T.Parallel(64, prefer_async=True):
                B[i] = A[i]
            for j in T.Unroll(4, unroll_factor=2):
                B[j] = A[j]

    annotations = {}

    def visit(node):
        if isinstance(node, tirx.For):
            annotations.update({str(key): value for key, value in node.annotations.items()})

    post_order_visit(loops.body, visit)
    assert bool(annotations["parallel_prefer_async"])
    assert int(annotations["pragma_unroll_factor"]) == 2


def test_ppu_reduce_nan_hint_is_recorded():
    @T.prim_func
    def reductions():
        with T.Kernel(1):
            src = T.alloc_local((4,), "float16")
            dst = T.alloc_local((1,), "float16")
            T.reduce_max(src, dst, nan_propagate=True)
            T.reduce_min(src, dst, nan_propagate=True)
            T.reduce_absmax(src, dst, nan_propagate=True)

    calls = []
    post_order_visit(
        reductions.body,
        lambda node: calls.append(node) if isinstance(node, tirx.Call) and getattr(node.op, "name", "") == "tl.tileop.reduce" else None,
    )
    assert len(calls) == 3
    assert all(bool(call.annotations["nan_propagate"]) for call in calls)
