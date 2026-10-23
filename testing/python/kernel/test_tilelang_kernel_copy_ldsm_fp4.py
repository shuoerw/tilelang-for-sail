"""Test PPU LowerLDSM for float4_e2m1fn T.copy fragments."""

import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.transform import PassConfigKey
from tilelang.layout import (
    make_full_bank_swizzled_layout,
    make_linear_layout,
)

DTYPE = "float4_e2m1fn"


def _fragment_layout(matrix):
    def a_layout(i, j):
        half_col = j // 2
        return (
            4 * (i % 8) + ((half_col % 16) // 4),
            16 * (j // 32) + 8 * (i // 8) + 2 * (half_col % 4) + (j % 2),
        )

    def b_layout(i, j):
        half_col = j // 2
        return (
            4 * (i % 8) + ((half_col % 16) // 4),
            16 * (i // 8) + 8 * (j // 32) + 2 * (half_col % 4) + (j % 2),
        )

    if matrix == "A":
        return T.Fragment((16, 64), forward_fn=a_layout)
    return T.Fragment((16, 64), forward_fn=b_layout)


def _make_kernel(matrix, swizzled):
    shared_shape = (16, 256 if swizzled else 64)
    fragment_shape = (16, 64)

    @T.prim_func
    def main(A: T.Tensor(shared_shape, DTYPE), B: T.Tensor(fragment_shape, DTYPE)):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared(shared_shape, DTYPE)
            fragment = T.alloc_fragment(fragment_shape, DTYPE)
            T.annotate_layout(
                {
                    shared: make_full_bank_swizzled_layout(shared) if swizzled else make_linear_layout(shared),
                    fragment: _fragment_layout(matrix),
                }
            )
            T.copy(A, shared)
            T.copy(shared[:, :64], fragment)
            T.copy(fragment, B)

    return main


def _run(matrix, swizzled):
    kernel = tilelang.compile(
        _make_kernel(matrix, swizzled),
        out_idx=[-1],
        pass_configs={PassConfigKey.TL_DISABLE_AIU_LOWER: False},
    )
    source = kernel.get_kernel_source()
    if swizzled:
        assert "tl::tix_ldmatrix_swzl_x4(" in source
    else:
        assert "tl::tix_ldmatrix_x4(" in source

    generator = torch.Generator(device="cuda").manual_seed(0)
    packed = torch.randint(0, 256, (16, 128), generator=generator, device="cuda", dtype=torch.uint8)
    data = packed if swizzled else packed[:, :32].contiguous()
    out = kernel(data.view(torch.int8))
    expected = data[:, :32] if swizzled else data
    torch.testing.assert_close(out.view(torch.uint8), expected.view(torch.uint8), rtol=0, atol=0)


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_fp4_linear_ldsm_a_and_b():
    _run("A", swizzled=False)
    _run("B", swizzled=False)


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_fp4_swzl_ldsm_a_and_b():
    _run("A", swizzled=True)
    _run("B", swizzled=True)


if __name__ == "__main__":
    tilelang.testing.main()
