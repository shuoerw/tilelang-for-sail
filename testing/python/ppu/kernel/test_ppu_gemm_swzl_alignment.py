"""Keep SWZL operand bases aligned independently of preceding allocations."""

import re

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.layout import make_full_bank_swizzled_layout, make_half_bank_swizzled_layout


@tilelang.testing.requires_ppu_compute_version(1, 5)
@pytest.mark.parametrize("dtype", ["bfloat16", "float8_e4m3fn"])
@pytest.mark.parametrize("row_bytes", [64, 128])
@pytest.mark.parametrize("prefix", [16, 64, 192])
@pytest.mark.parametrize("scope", ["shared.dyn", "shared"])
def test_gemm_swzl_alignment(dtype, row_bytes, prefix, scope):
    k = row_bytes * 8 // tilelang.tvm.DataType(dtype).bits
    layout = make_full_bank_swizzled_layout if row_bytes == 128 else make_half_bank_swizzled_layout

    @T.prim_func
    def main(
        A: T.Tensor((64, k), dtype),
        B: T.Tensor((64, k), dtype),
        P: T.Tensor((prefix,), "uint8"),
        C: T.Tensor((64, 64), "float32"),
        Q: T.Tensor((prefix,), "uint8"),
    ):
        with T.Kernel(1, threads=128):
            a_pad = T.alloc_shared((prefix,), "uint8", scope=scope)
            b_A = T.alloc_shared((64, k), dtype, scope=scope)
            c_B = T.alloc_shared((64, k), dtype, scope=scope)
            acc = T.alloc_fragment((64, 64), "float32")
            T.annotate_layout({b_A: layout(b_A), c_B: layout(c_B)})
            T.copy(P, a_pad)
            T.copy(A, b_A)
            T.copy(B, c_B)
            T.clear(acc)
            T.gemm(b_A, c_B, acc, transpose_B=True)
            T.copy(acc, C)
            # Keep the unrelated small allocation live across the GEMM.
            T.copy(a_pad, Q)

    torch.manual_seed(44)
    a = torch.randn(64, k, device="cuda").to(getattr(torch, dtype))
    b = torch.randn(64, k, device="cuda").to(getattr(torch, dtype))
    p = torch.randint(0, 255, (prefix,), device="cuda", dtype=torch.uint8)
    for disabled in (False, True):
        kernel = tilelang.compile(
            main,
            out_idx=[3, 4],
            pass_configs={
                "tl.disable_aiu_lower": True,
                "tl.disable_ldmat_swzl": disabled,
            },
        )
        source = kernel.get_kernel_source()
        assert ("tix_ldmatrix_swzl" in source) == (not disabled)
        alignment = 16 if disabled else row_bytes
        if scope == "shared":
            assert re.search(r"__align__\(16\).*\ba_pad\[", source), source
            for name in ("b_A", "c_B"):
                assert re.search(rf"__align__\({alignment}\).*\b{name}\[", source), source
        else:
            base = "buf_dyn_shmem"
            assert re.search(rf"__align__\(1024\).*\b{base}\[", source), source
            offsets = {}
            for name in ("a_pad", "b_A", "c_B"):
                match = re.search(rf"void\*\s+{name}\s*=.*?{base}\s*\+\s*(\d+)", source)
                assert match, source
                offsets[name] = int(match.group(1))
            assert offsets["a_pad"] == 0
            assert offsets["b_A"] == (prefix + alignment - 1) // alignment * alignment
            assert offsets["c_B"] % alignment == 0
        c, q = kernel(a, b, p)
        torch.testing.assert_close(c, a.float() @ b.float().T, atol=0.02, rtol=0.02)
        torch.testing.assert_close(q, p, atol=0, rtol=0)


if __name__ == "__main__":
    tilelang.testing.main()
