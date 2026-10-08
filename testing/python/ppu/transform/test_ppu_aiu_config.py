"""Native PassContext and Python normalization agree on AIU configuration."""

import pytest
import tilelang
import tilelang.language as T
import tilelang.ppu.transform
from tilelang import tvm
from tilelang.layout import make_full_bank_swizzled_layout
from tilelang.transform.pass_config import normalize_pass_configs

DISABLE_AIU = "tl.disable_aiu_lower"
CONFIGS = [
    ({}, False),
    ({DISABLE_AIU: False}, True),
    ({DISABLE_AIU: True}, False),
]


@pytest.mark.parametrize("config, expected", CONFIGS)
def test_direct_pass_context_honors_aiu_disable_flag(config, expected):
    target = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})

    @T.prim_func
    def main(A: T.Tensor((16, 128), "float16")):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((16, 128), "float16")
            T.annotate_layout({shared: make_full_bank_swizzled_layout(shared)})
            T.copy(A, shared, prefer_instruction="aiu")

    # Construct the native context directly: no Python config normalization.
    with target, tvm.transform.PassContext(config=config):
        mod = tvm.tirx.transform.BindTarget(target)(tvm.IRModule.from_expr(main))
        mod = tilelang.transform.MaterializeKernelLaunch()(mod)
        mod = tilelang.ppu.transform.LayoutInference()(mod)
        mod = tilelang.ppu.transform.LowerTileOp()(mod)
    calls = []
    tvm.tirx.stmt_functor.post_order_visit(
        mod["main"].body, lambda n: calls.append(n.op.name) if isinstance(n, tvm.tirx.Call) and isinstance(n.op, tvm.ir.Op) else None
    )
    assert ("tl.ppu_aiu_load" in calls) == expected
    assert normalize_pass_configs(config) == config
