import pytest


@pytest.mark.parametrize("arch", ["ppu_10", "ppu_15"])
@pytest.mark.parametrize("pass_name", ["LayoutInference", "LowerTileOp"])
@pytest.mark.parametrize(
    "m,n,num_warps,policy,expected,error",
    [
        (32, 32, 4, 0, (2, 2), None),
        (16, 104, 13, 0, (1, 13), None),
        (64, 64, 8, 1, (4, 2), None),
        (128, 32, 8, 2, (2, 4), None),
        (128, 48, 8, 1, (8, 1), None),
        (48, 64, 8, 2, (1, 8), None),
        (16, 8, 1, 0, (1, 1), None),
        (16, 104, 12, 0, None, "warp partition would truncate output tile"),
        (288, 8, 17, 0, None, "warp partition would truncate output tile"),
        (16, 104, 12, 1, None, "warp partition would truncate output tile"),
        (336, 8, 20, 2, None, "warp partition would truncate output tile"),
        (48, 48, 8, 0, None, "warp partition would truncate output tile"),
        (48, 64, 8, 1, None, "m_warp \\* n_warp must equal num_warps"),
        (128, 48, 8, 2, None, "m_warp \\* n_warp must equal num_warps"),
        (48, 48, 8, 1, None, "m_warp \\* n_warp must equal num_warps"),
        (16, 8, 8, 0, None, "m_warp \\* n_warp must equal num_warps"),
    ],
)
def test_ppu_sparse_gemm_registry_and_partition(m, n, num_warps, policy, expected, error, arch, pass_name, monkeypatch):
    import tilelang.ppu.language as T
    import tilelang.ppu.transform
    from tilelang import tvm
    from tilelang.ppu.op.gemm_sp import gemm_sp_mma

    target = tvm.target.Target({"kind": "ppu", "arch": arch})
    warp_policy = T.GemmWarpPolicy(policy)
    emitters = []
    make_emitter = gemm_sp_mma.PPUSparseTensorCoreIntrinEmitter

    def record_emitter(*args, **kwargs):
        emitter = make_emitter(*args, **kwargs)
        emitters.append(emitter)
        return emitter

    monkeypatch.setattr(gemm_sp_mma, "PPUSparseTensorCoreIntrinEmitter", record_emitter)

    @T.prim_func
    def main():
        T.func_attr({"global_symbol": "main", "target": target})
        T.launch_thread("blockIdx.x", 1)
        T.launch_thread("threadIdx.x", num_warps * 32)
        with T.sblock("root"):
            a_shared = T.sblock_alloc_buffer((m, 16), "float16", scope="shared")
            e_shared = T.sblock_alloc_buffer((m, 2), "int16", scope="shared")
            b_shared = T.sblock_alloc_buffer((32, n), "float16", scope="shared")
            c_fragment = T.sblock_alloc_buffer((m, n), "float32", scope="local.fragment")
            T.gemm_sp(a_shared, e_shared, b_shared, c_fragment, policy=warp_policy, clear_accum=True)

    mod = tvm.IRModule.from_expr(main)
    with target:
        if error is not None:
            with pytest.raises(Exception, match=error):
                getattr(tilelang.ppu.transform, pass_name)()(mod)
            return
        mod = tilelang.ppu.transform.LayoutInference()(mod)
        mod = tilelang.ppu.transform.LowerTileOp()(mod)
    assert emitters
    for emitter in emitters:
        assert (emitter.block_row_warps, emitter.block_col_warps) == expected
        assert m % (emitter.block_row_warps * 16) == 0
        assert n % (emitter.block_col_warps * 8) == 0
    calls = []
    tvm.tirx.stmt_functor.post_order_visit(
        mod["main"].body,
        lambda node: calls.append(node.op.name) if isinstance(node, tvm.tirx.Call) and isinstance(node.op, tvm.ir.Op) else None,
    )
    assert "tirx.ptx_mma_sp" in calls
    assert "tl.tileop.gemm_sp" not in calls
