"""Copy preferences must respect memory direction, dtype and async semantics."""

import pytest
import tilelang
import tilelang.ppu.language as T
import tilelang.ppu.transform
from tilelang import tvm
from tilelang.layout import make_full_bank_swizzled_layout, make_gemm_fragment_8x8, make_linear_layout

PPU_TARGET = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})


def _lower(program, enabled=True):
    mod = tvm.IRModule.from_expr(program)
    with PPU_TARGET, tvm.transform.PassContext(config={"tl.disable_aiu_lower": not enabled}):
        mod = tvm.tirx.transform.BindTarget(PPU_TARGET)(mod)
        mod = tilelang.transform.MaterializeKernelLaunch()(mod)
        mod = tilelang.ppu.transform.LayoutInference()(mod)
        mod = tilelang.ppu.transform.LowerTileOp()(mod)
    ops = []
    tvm.tirx.stmt_functor.post_order_visit(
        mod["main"].body, lambda n: ops.append(n.op.name) if isinstance(n, tvm.tirx.Call) and isinstance(n.op, tvm.ir.Op) else None
    )
    return ops


def test_sync_aiu_preference_keeps_shared_to_fragment_ldsm():
    layout = make_gemm_fragment_8x8().repeat([2, 2], repeat_on_thread=False)

    @T.prim_func
    def main(A: T.Tensor((16, 16), "float16"), B: T.Tensor((16, 16), "float16")):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((16, 16), "float16")
            fragment = T.alloc_fragment((16, 16), "float16")
            T.annotate_layout({shared: make_linear_layout(shared), fragment: layout})
            T.copy(A, shared)
            T.copy(shared, fragment, prefer_instruction="aiu")
            T.copy(fragment, B)

    ops = _lower(main)
    assert "tl.ptx_ldmatrix" in ops
    assert "tl.ppu_aiu_load" not in ops


@pytest.mark.parametrize("direction", ["shared_to_shared", "shared_to_global", "dtype_mismatch"])
def test_unsupported_async_copy_has_direct_diagnostic(direction):
    dtype = "float32" if direction == "dtype_mismatch" else "float16"

    @T.prim_func
    def main(A: T.Tensor((16, 64), "float16"), B: T.Tensor((16, 64), "float16")):
        with T.Kernel(1, threads=128):
            source = T.alloc_shared((16, 64), "float16")
            destination = T.alloc_shared((16, 64), dtype)
            T.copy(A, source)
            if direction == "shared_to_shared":
                T.async_copy(source, destination)
            elif direction == "shared_to_global":
                T.async_copy(source, B)
            else:
                T.async_copy(A, destination)

    diagnostic = "equal byte-addressable dtypes" if direction == "dtype_mismatch" else "only supports global->shared"
    with pytest.raises(tvm.error.InternalError, match=diagnostic):
        _lower(main)


def test_aiu_ineligible_dtype_uses_cp_async():
    @T.prim_func
    def main(A: T.Tensor((16, 64), "float32")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((16, 64), "float32")
            T.copy(A, shared, prefer_instruction="aiu")

    ops = _lower(main)
    assert "tl.ppu_aiu_load" not in ops
    assert "tl.ptx_cp_async" in ops


@pytest.mark.parametrize("prefer_instruction", ["cp_async", "sync"])
def test_explicit_copy_preference_prevents_automatic_aiu(prefer_instruction):
    @T.prim_func
    def main(A: T.Tensor((16, 64), "float16")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((16, 64), "float16")
            T.annotate_layout({shared: make_full_bank_swizzled_layout(shared)})
            T.copy(A, shared, prefer_instruction=prefer_instruction)

    ops = _lower(main)
    assert "tl.ppu_aiu_load" not in ops
    assert ("tl.ptx_cp_async" in ops) == (prefer_instruction == "cp_async")


@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("annotated_aiu", [False, True])
@pytest.mark.parametrize("no_implicit_commit_wait", [False, True])
def test_async_copy_keeps_cp_async_semantics(enabled, annotated_aiu, no_implicit_commit_wait):
    annotations = {}
    if annotated_aiu:
        annotations["prefer_instruction"] = tvm.tirx.StringImm("aiu")
    if no_implicit_commit_wait:
        annotations["no_implicit_async_commit_wait"] = tvm.tirx.IntImm("int32", 1)

    @T.prim_func
    def main(A: T.Tensor((16, 64), "float16")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((16, 64), "float16")
            T.annotate_layout({shared: make_full_bank_swizzled_layout(shared)})
            T.async_copy(A, shared, annotations=annotations)

    ops = _lower(main, enabled=enabled)
    assert "tl.ppu_aiu_load" not in ops
    assert "tl.ptx_cp_async" in ops
    assert ops.count("tirx.ptx_commit_group") == (0 if no_implicit_commit_wait else 1)
    assert "tirx.ptx_wait_group" not in ops


@pytest.mark.parametrize("direction", ["dtype_conversion", "shared_to_shared", "shared_to_global", "global_to_global"])
@pytest.mark.parametrize("enabled", [False, True])
@pytest.mark.parametrize("pipeline_managed", [False, True])
def test_aiu_preference_does_not_enable_generic_fp4_fallback(direction, enabled, pipeline_managed):
    dtype = "float16" if direction == "dtype_conversion" else "float4_e2m1fn"
    annotations = {"no_implicit_async_commit_wait": 1} if pipeline_managed else {}

    @T.prim_func
    def main(A: T.Tensor((16, 512), dtype), B: T.Tensor((16, 512), "float4_e2m1fn")):
        with T.Kernel(1, threads=128):
            source = T.alloc_shared((16, 512), "float4_e2m1fn")
            destination = T.alloc_shared((16, 512), "float4_e2m1fn")
            if direction == "dtype_conversion":
                T.copy(A, destination, prefer_instruction="aiu", annotations=annotations)
            elif direction == "shared_to_shared":
                T.copy(source, destination, prefer_instruction="aiu", annotations=annotations)
            elif direction == "shared_to_global":
                T.copy(source, B, prefer_instruction="aiu", annotations=annotations)
            else:
                T.copy(A, B, prefer_instruction="aiu", annotations=annotations)

    with pytest.raises(tvm.error.InternalError, match="AIU preference cannot fall back for this FP4 copy"):
        _lower(main, enabled=enabled)


def test_fp4_aiu_preference_reports_disabled_aiu():
    @T.prim_func
    def main(A: T.Tensor((16, 512), "float4_e2m1fn")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((16, 512), "float4_e2m1fn")
            T.annotate_layout({shared: make_full_bank_swizzled_layout(shared)})
            T.copy(A, shared, prefer_instruction="aiu")

    with pytest.raises(tvm.error.InternalError, match="AIU preference cannot fall back for this FP4 copy"):
        _lower(main, enabled=False)
