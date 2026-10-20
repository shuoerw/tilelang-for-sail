"""Host-only regressions for PPU warp partition compatibility and truncation checks."""

import pytest

import tilelang.ppu.language as T
import tilelang.ppu.transform
import tilelang.testing
from tilelang import tvm
from tilelang.ppu.op.gemm.gemm_mma import PPUGemmMMA
from tvm.tirx.stmt_functor import post_order_visit


_POLICIES = [T.GemmWarpPolicy.Square, T.GemmWarpPolicy.FullRow, T.GemmWarpPolicy.FullCol]


def _gemm_module(m, n, num_warps, policy, target):
    @T.prim_func
    def main():
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        T.launch_thread("threadIdx.x", num_warps * 32)
        with T.sblock("root"):
            a_shared = T.sblock_alloc_buffer((m, 32), "float16", scope="shared")
            b_shared = T.sblock_alloc_buffer((32, n), "float16", scope="shared")
            c_fragment = T.sblock_alloc_buffer((m, n), "float32", scope="local.fragment")
            T.gemm(a_shared, b_shared, c_fragment, policy=policy, clear_accum=True)

    return tvm.IRModule.from_expr(main)


def _apply_pass(mod, target, pass_name):
    with target:
        return getattr(tilelang.ppu.transform, pass_name)()(mod)


@pytest.mark.parametrize("pass_name", ["LayoutInference", "LowerTileOp"])
@pytest.mark.parametrize(
    "m,n,num_warps,policy",
    [
        # Square formerly rounded N/12 and M/17 down to a 16-wide tile.
        (16, 208, 12, T.GemmWarpPolicy.Square),
        (288, 16, 17, T.GemmWarpPolicy.Square),
        # FullRow/FullCol formerly chose the requested nondividing warp count.
        (16, 208, 12, T.GemmWarpPolicy.FullRow),
        (208, 16, 12, T.GemmWarpPolicy.FullRow),
        (336, 16, 20, T.GemmWarpPolicy.FullCol),
        (16, 336, 20, T.GemmWarpPolicy.FullCol),
        (48, 48, 8, T.GemmWarpPolicy.FullRow),
        (48, 48, 8, T.GemmWarpPolicy.FullCol),
        (48, 64, 4, T.GemmWarpPolicy.Square),
        (48, 80, 8, T.GemmWarpPolicy.Square),
    ],
)
def test_ppu_gemm_uncovered_partition_rejected(m, n, num_warps, policy, pass_name):
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})
    mod = _gemm_module(m, n, num_warps, policy, target)
    with pytest.raises(Exception, match="warp partition would truncate output tile") as exc_info:
        _apply_pass(mod, target, pass_name)
    message = str(exc_info.value)
    assert f"M={m}" in message
    assert f"N={n}" in message
    assert f"num_warps={num_warps}" in message
    assert f"policy={policy.name}" in message
    assert "M must be divisible by" in message


@pytest.mark.parametrize("pass_name", ["LayoutInference", "LowerTileOp"])
@pytest.mark.parametrize("policy", _POLICIES)
@pytest.mark.parametrize("m,n,axis", [(8, 16, "M"), (16, 8, "N")])
def test_ppu_gemm_requires_16x16_atoms(m, n, axis, policy, pass_name):
    # In particular, CUDA's eight-column rule must not leak into PPU.
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})
    mod = _gemm_module(m, n, 1, policy, target)
    with pytest.raises(Exception, match=f"{axis} must be divisible by 16"):
        _apply_pass(mod, target, pass_name)


@pytest.mark.parametrize("arch", ["ppu_10", "ppu_15"])
@pytest.mark.parametrize(
    "m,n,num_warps,policy,expected",
    [
        (16, 208, 13, T.GemmWarpPolicy.Square, (1, 13)),
        (288, 16, 18, T.GemmWarpPolicy.Square, (18, 1)),
        (64, 96, 4, T.GemmWarpPolicy.Square, (2, 2)),
        (48, 128, 8, T.GemmWarpPolicy.FullRow, (3, 2)),
        (128, 48, 8, T.GemmWarpPolicy.FullCol, (2, 3)),
        (128, 192, 12, T.GemmWarpPolicy.FullRow, (8, 1)),
        (192, 128, 12, T.GemmWarpPolicy.FullCol, (1, 8)),
        (48, 64, 8, T.GemmWarpPolicy.FullRow, (3, 2)),
        (64, 48, 8, T.GemmWarpPolicy.FullCol, (2, 3)),
        (48, 48, 8, T.GemmWarpPolicy.Square, (1, 1)),
        (32, 48, 8, T.GemmWarpPolicy.FullRow, (2, 3)),
        (32, 48, 8, T.GemmWarpPolicy.FullCol, (2, 3)),
        (32, 48, 8, T.GemmWarpPolicy.Square, (1, 1)),
        (16, 16, 7, T.GemmWarpPolicy.Square, (1, 1)),
        (16, 16, 7, T.GemmWarpPolicy.FullRow, (1, 1)),
        (16, 16, 7, T.GemmWarpPolicy.FullCol, (1, 1)),
        (32, 48, 6, T.GemmWarpPolicy.Square, (2, 3)),
    ],
)
def test_ppu_gemm_covering_partition_lowers(m, n, num_warps, policy, expected, arch, monkeypatch):
    target = tvm.target.Target({"kind": "ppu", "arch": arch})
    emitters = []
    make_emitter = PPUGemmMMA._make_mma_emitter

    def record_emitter(self, *args, **kwargs):
        # Observe the real native partition and emitter; do not replace either.
        emitter = make_emitter(self, *args, **kwargs)
        emitters.append(emitter)
        return emitter

    monkeypatch.setattr(PPUGemmMMA, "_make_mma_emitter", record_emitter)
    mod = _gemm_module(m, n, num_warps, policy, target)
    inferred = _apply_pass(mod, target, "LayoutInference")
    inference_count = len(emitters)
    assert inference_count > 0
    lowered = _apply_pass(inferred, target, "LowerTileOp")
    assert len(emitters) > inference_count

    for emitter in emitters:
        m_warp, n_warp = emitter.block_row_warps, emitter.block_col_warps
        assert (m_warp, n_warp) == expected
        assert 0 < m_warp * n_warp <= num_warps
        assert m % (m_warp * 16) == 0
        assert n % (n_warp * 16) == 0
        assert m_warp * emitter.warp_rows * 16 == m
        assert n_warp * emitter.warp_cols * 16 == n
        assert emitter.threads == m_warp * n_warp * 32

    mma_calls = []
    tile_calls = []

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.Op):
            if node.op.name == "tirx.ptx_mma":
                mma_calls.append(node)
            elif node.op.name == "tl.tileop.gemm":
                tile_calls.append(node)

    post_order_visit(lowered["main"].body, visit)
    assert mma_calls
    assert not tile_calls
    assert all(str(call.args[0].value) == "m16n16k16" for call in mma_calls)


if __name__ == "__main__":
    tilelang.testing.main()
