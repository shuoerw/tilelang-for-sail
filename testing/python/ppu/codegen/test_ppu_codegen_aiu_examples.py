"""AIU-enabled examples must emit AIU loads without requiring a device."""

from pathlib import Path
import runpy

import pytest

from tilelang import tvm
from tilelang.engine.lower import lower


TARGET = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})
EXAMPLES = Path(__file__).resolve().parents[4] / "examples" / "ppu"
GEMM_ARGS = dict(M=256, N=256, K=256, block_M=64, block_N=64, block_K=64, num_stages=2)


@pytest.mark.parametrize(
    "path, entry, kwargs",
    [
        ("gemm/ppu_example_gemm.py", "matmul", GEMM_ARGS),
        (
            "gemm_fp8/example_tilelang_gemm_fp8.py",
            "matmul",
            {k: v for k, v in GEMM_ARGS.items() if k != "num_stages"} | {"dtype": "float8_e4m3fn"},
        ),
        (
            "gemm_fp4/ppu_example_gemm_fp4.py",
            "matmul",
            GEMM_ARGS | {"block_K": 128, "trans_A": False, "trans_B": True},
        ),
        (
            "gemm_fp4/ppu_example_mxfp4_scaled_gemm.py",
            "matmul_scaled",
            GEMM_ARGS | {"block_K": 128},
        ),
        (
            "flash_attention/ppu_example_mha_fwd_bshd.py",
            "flashattn",
            dict(batch=1, heads=4, seq_len=128, dim=64, is_causal=False, num_stages=2),
        ),
    ],
)
def test_aiu_enabled_examples_select_automatically(monkeypatch, path, entry, kwargs):
    example = EXAMPLES / path
    monkeypatch.syspath_prepend(str(example.parent))
    with TARGET:
        kernel = runpy.run_path(str(example))[entry]
        kernel = getattr(kernel, "jit_impl", kernel)
        program = kernel.get_tir(**kwargs)
        with tvm.transform.PassContext(opt_level=3, config=kernel.pass_configs):
            source = lower(program, target=TARGET).kernel_source
    assert "tl::aiu_load(" in source, f"{path} did not generate AIU loads"
