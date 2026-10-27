"""Tests for proven PPU scalar-to-packed subword warp-shuffle rewrites."""

import tilelang.ppu.language as T
import tilelang.ppu.transform
import tilelang.testing
from tilelang import tvm
from tvm import s_tir
from tvm.tirx.stmt_functor import post_order_visit


def _make_relayout(
    lane_stride: int = 1,
    dtype=T.float8_e4m3fn,
    arch="ppu_15",
    dst_offset: int = 0,
    false_width: int = 32,
):
    target = tvm.target.Target({"kind": "ppu", "arch": arch})

    @T.prim_func
    def main():
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        T.launch_thread("threadIdx.x", 32)
        src = T.alloc_buffer((16,), T.float32, scope="local")
        dst = T.alloc_buffer((16 + dst_offset,), dtype, scope="local")
        lane = T.get_lane_idx()
        for i in T.unroll(16):
            source_lane = lane // 4 * 4 + lane % 2 * 2 + i % 4 // 2 * lane_stride
            low = T.shfl_sync(src[i // 8 * 8 + i % 8 // 4 * 2 + i % 2], source_lane)
            high = T.shfl_sync(
                src[i // 8 * 8 + i % 8 // 4 * 2 + i % 2 + 4],
                source_lane,
                false_width,
            )
            dst[i + dst_offset] = T.cast(T.if_then_else(lane % 4 < 2, low, high), dtype)

    return tvm.IRModule.from_expr(main)


def _make_unconditional_relayout(dtype, arch, source_lane_count=1, dst_offset=0):
    target = tvm.target.Target({"kind": "ppu", "arch": arch})
    source_dtype = T.int32 if dtype == T.int8 else T.float32

    @T.prim_func
    def main():
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        T.launch_thread("threadIdx.x", 32)
        src = T.alloc_buffer((16,), source_dtype, scope="local")
        dst = T.alloc_buffer((16 + dst_offset,), dtype, scope="local")
        lane = T.get_lane_idx()
        for i in T.unroll(16):
            source_lane = lane // 4 * 4 + (i % 4) % source_lane_count
            dst[i + dst_offset] = T.cast(
                T.shfl_sync(src[i], source_lane),
                dtype,
            )

    return tvm.IRModule.from_expr(main)


def _make_conditional_lane_relayout(
    dtype,
    arch,
    source_lane_count,
    dst_offset=0,
    false_width=32,
    false_source_lane_count=None,
):
    target = tvm.target.Target({"kind": "ppu", "arch": arch})
    source_dtype = T.int32 if dtype == T.int8 else T.float32
    false_count = false_source_lane_count or source_lane_count

    @T.prim_func
    def main():
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        T.launch_thread("threadIdx.x", 32)
        src = T.alloc_buffer((32,), source_dtype, scope="local")
        dst = T.alloc_buffer((16 + dst_offset,), dtype, scope="local")
        lane = T.get_lane_idx()
        for i in T.unroll(16):
            source_lane = lane // 4 * 4 + (i % 4) % source_lane_count
            false_lane = lane // 4 * 4 + (i % 4) % false_count
            true_value = T.shfl_sync(src[i], source_lane)
            false_value = T.shfl_sync(src[i + 16], false_lane, false_width)
            dst[i + dst_offset] = T.cast(T.if_then_else(lane % 2 == 0, true_value, false_value), dtype)

    return tvm.IRModule.from_expr(main)


def _collect(mod):
    calls = {}
    loops = []
    stores = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.Op):
            name = str(node.op.name)
            calls[name] = calls.get(name, 0) + 1
        elif isinstance(node, tvm.tirx.For):
            loops.append(node)
        elif isinstance(node, tvm.tirx.BufferStore):
            stores.append(node)

    post_order_visit(mod["main"].body, visit)
    return calls, loops, stores


def _effective_unrolled_count(static_count, loops):
    count = static_count
    for loop in loops:
        count *= int(loop.extent)
    return count


def test_pack_fp8_warp_shuffle_e4m3fn_fast_path():
    mod = tilelang.ppu.transform.PackSubwordWarpShuffle()(_make_relayout())
    calls, loops, stores = _collect(mod)

    assert calls.get("tl.shfl_sync", 0) == 2
    assert calls.get("tirx.call_pure_extern", 0) == 1
    assert any(int(loop.extent) == 4 for loop in loops)
    assert any(store.value.dtype.lanes == 4 for store in stores)


def test_pack_fp8_warp_shuffle_after_pipeline_unroll():
    # Reproduce the exact three passes immediately before this rewrite in the
    # production PPU pipeline, and cover its resulting SeqStmt form.
    mod = tilelang.transform.UnrollLoop()(_make_relayout())
    mod = s_tir.transform.RenormalizeSplitPattern()(mod)
    mod = tvm.tirx.transform.Simplify()(mod)
    mod = tilelang.ppu.transform.PackSubwordWarpShuffle()(mod)
    calls, loops, stores = _collect(mod)

    assert _effective_unrolled_count(calls.get("tl.shfl_sync", 0), loops) == 8
    assert _effective_unrolled_count(calls.get("tirx.call_pure_extern", 0), loops) == 4
    assert _effective_unrolled_count(len(stores), loops) == 4
    assert all(store.value.dtype.lanes == 4 for store in stores)


def test_bf16x2_unconditional_source_lane_remains_scalar_on_ppu15():
    mod = tilelang.ppu.transform.PackSubwordWarpShuffle()(_make_unconditional_relayout(T.bfloat16, "ppu_15"))
    calls, loops, stores = _collect(mod)

    assert calls.get("tl.shfl_sync", 0) == 1
    assert any(int(loop.extent) == 16 for loop in loops)
    assert all(store.value.dtype.lanes == 1 for store in stores)


def test_pack_int8x4_single_source_lane_on_ppu10():
    mod = tilelang.ppu.transform.PackSubwordWarpShuffle()(_make_unconditional_relayout(T.int8, "ppu_10"))
    calls, loops, stores = _collect(mod)

    assert calls.get("tl.shfl_sync", 0) == 1
    assert any(int(loop.extent) == 4 for loop in loops)
    assert any(str(store.value.dtype) == "int8x4" for store in stores)


def test_pack_int8x4_single_source_lane_after_pipeline_unroll():
    mod = tilelang.transform.UnrollLoop()(_make_unconditional_relayout(T.int8, "ppu_10"))
    mod = s_tir.transform.RenormalizeSplitPattern()(mod)
    mod = tvm.tirx.transform.Simplify()(mod)
    mod = tilelang.ppu.transform.PackSubwordWarpShuffle()(mod)
    calls, loops, stores = _collect(mod)

    assert _effective_unrolled_count(calls.get("tl.shfl_sync", 0), loops) == 4
    assert _effective_unrolled_count(len(stores), loops) == 4
    assert all(str(store.value.dtype) == "int8x4" for store in stores)


def test_unconditional_shuffle_admission():
    for dtype, arch in (
        (T.float8_e4m3fn, "ppu_15"),
        (T.int8, "ppu_10"),
        (T.int8, "ppu_15"),
    ):
        for source_lane_count in (1, 2, 3, 4):
            mod = tilelang.ppu.transform.PackSubwordWarpShuffle()(_make_unconditional_relayout(dtype, arch, source_lane_count))
            calls, loops, stores = _collect(mod)

            packed = dtype == T.float8_e4m3fn or source_lane_count <= 2
            assert calls.get("tl.shfl_sync", 0) == (source_lane_count if packed else 1)
            assert calls.get("tirx.call_pure_extern", 0) == (source_lane_count - 1 if packed else 0)
            assert any(int(loop.extent) == (4 if packed else 16) for loop in loops)
            assert any(store.value.dtype.lanes == (4 if packed else 1) for store in stores)


def test_conditional_shuffle_admission():
    for dtype, arch in (
        (T.float8_e4m3fn, "ppu_15"),
        (T.int8, "ppu_10"),
        (T.int8, "ppu_15"),
    ):
        for source_lane_count in (1, 2, 3, 4):
            mod = tilelang.ppu.transform.PackSubwordWarpShuffle()(_make_conditional_lane_relayout(dtype, arch, source_lane_count))
            calls, loops, stores = _collect(mod)

            packed = dtype == T.float8_e4m3fn or source_lane_count <= 2
            assert calls.get("tl.shfl_sync", 0) == (2 * source_lane_count if packed else 2)
            assert calls.get("tirx.call_pure_extern", 0) == (2 * (source_lane_count - 1) if packed else 0)
            assert any(int(loop.extent) == (4 if packed else 16) for loop in loops)
            assert any(store.value.dtype.lanes == (4 if packed else 1) for store in stores)
            if not packed:
                binds = []

                def collect_bind(node, binds=binds):
                    if isinstance(node, tvm.tirx.Bind):
                        binds.append(node)

                post_order_visit(mod["main"].body, collect_bind)
                assert len(binds) >= 2


def test_conditional_shuffle_rejects_three_lanes_in_either_candidate():
    for true_count, false_count in ((2, 3), (3, 2)):
        mod = tilelang.ppu.transform.PackSubwordWarpShuffle()(
            _make_conditional_lane_relayout(
                T.int8,
                "ppu_15",
                true_count,
                false_source_lane_count=false_count,
            )
        )
        calls, loops, stores = _collect(mod)
        assert calls.get("tl.shfl_sync", 0) == 2
        assert calls.get("tirx.call_pure_extern", 0) == 0
        assert any(int(loop.extent) == 16 for loop in loops)
        assert all(store.value.dtype.lanes == 1 for store in stores)


def test_pack_fp8_warp_shuffle_non_adjacent_lanes_uses_general_gather():
    mod = tilelang.ppu.transform.PackSubwordWarpShuffle()(_make_relayout(lane_stride=2))
    calls, loops, stores = _collect(mod)

    assert calls.get("tl.shfl_sync", 0) == 4
    assert calls.get("tirx.call_pure_extern", 0) == 2
    assert any(int(loop.extent) == 4 for loop in loops)
    assert any(store.value.dtype.lanes == 4 for store in stores)


def test_pack_fp8_warp_shuffle_is_ppu15_only():
    mod = tilelang.ppu.transform.PackSubwordWarpShuffle()(_make_relayout(arch="ppu_10"))
    calls, loops, stores = _collect(mod)

    assert calls.get("tl.shfl_sync", 0) == 2
    assert calls.get("tirx.call_pure_extern", 0) == 0
    assert any(int(loop.extent) == 16 for loop in loops)
    assert all(store.value.dtype.lanes == 1 for store in stores)


def test_pack_fp8_warp_shuffle_does_not_widen_unproven_subword_types():
    # Only E4M3FN and signed INT8 pass the current profitability gate.
    for dtype in (
        T.float16,
        T.bfloat16,
        T.uint8,
        T.float8_e4m3,
        T.float8_e5m2,
    ):
        mod = tilelang.ppu.transform.PackSubwordWarpShuffle()(_make_relayout(dtype=dtype))
        calls, loops, stores = _collect(mod)

        assert calls.get("tl.shfl_sync", 0) == 2
        assert calls.get("tirx.call_pure_extern", 0) == 0
        assert any(int(loop.extent) == 16 for loop in loops)
        assert all(store.value.dtype.lanes == 1 for store in stores)


def test_pack_fp8_warp_shuffle_requires_aligned_destination():
    mod = tilelang.ppu.transform.PackSubwordWarpShuffle()(_make_relayout(dst_offset=1))
    calls, loops, stores = _collect(mod)

    assert calls.get("tl.shfl_sync", 0) == 2
    assert calls.get("tirx.call_pure_extern", 0) == 0
    assert any(int(loop.extent) == 16 for loop in loops)
    assert all(store.value.dtype.lanes == 1 for store in stores)


def test_pack_int8x4_source_lane_requires_aligned_destination():
    mod = tilelang.ppu.transform.PackSubwordWarpShuffle()(_make_unconditional_relayout(T.int8, "ppu_10", dst_offset=1))
    calls, loops, stores = _collect(mod)

    assert calls.get("tl.shfl_sync", 0) == 1
    assert any(int(loop.extent) == 16 for loop in loops)
    assert all(store.value.dtype.lanes == 1 for store in stores)


def test_packed_shuffle_requires_shared_control_for_select_branches():
    mod = tilelang.ppu.transform.PackSubwordWarpShuffle()(_make_relayout(false_width=16))
    calls, loops, stores = _collect(mod)

    assert calls.get("tl.shfl_sync", 0) == 2
    assert calls.get("tirx.call_pure_extern", 0) == 0
    assert any(int(loop.extent) == 16 for loop in loops)
    assert all(store.value.dtype.lanes == 1 for store in stores)


def test_unrolled_packed_shuffles_require_aligned_destination():
    cases = (
        _make_relayout(dst_offset=1),
        _make_unconditional_relayout(T.int8, "ppu_10", dst_offset=1),
    )
    for mod in cases:
        mod = tilelang.transform.UnrollLoop()(mod)
        mod = s_tir.transform.RenormalizeSplitPattern()(mod)
        mod = tvm.tirx.transform.Simplify()(mod)
        mod = tilelang.ppu.transform.PackSubwordWarpShuffle()(mod)
        _, loops, stores = _collect(mod)

        scalar_stores = [store for store in stores if store.value.dtype.lanes == 1]
        vector_stores = [store for store in stores if store.value.dtype.lanes == 4]
        if loops:
            assert scalar_stores
            assert not vector_stores
        else:
            assert any(int(store.indices[0]) == 1 for store in scalar_stores)
            assert all(int(store.indices[0].base) % 4 == 0 for store in vector_stores)


if __name__ == "__main__":
    tilelang.testing.main()
