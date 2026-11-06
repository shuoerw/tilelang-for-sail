"""Generated slots and numerical results for the PPU register pipeline."""

import tilelang
import tilelang.ppu.language as T
import tilelang.testing

from testing.python.ppu.kernel.test_ppu_kernel_gemm import matmul


@tilelang.testing.requires_ppu
def test_ppu_gemm_register_pingpong_slots():
    program = matmul(
        512,
        1024,
        768,
        128,
        128,
        32,
        False,
        False,
        "float16",
        "float16",
        "float32",
        0,
        128,
    )
    kernel = tilelang.compile(program, out_idx=[2])
    source = kernel.get_kernel_source()

    for name in ("A_local_0", "A_local_1", "B_local_0", "B_local_1"):
        assert name in source
    assert "register_pipeline_stage" not in source

    import torch

    generator = torch.Generator().manual_seed(7)
    a_cpu = torch.randn((512, 768), generator=generator, dtype=torch.float16)
    b_cpu = torch.randn((768, 1024), generator=generator, dtype=torch.float16)
    expected = (a_cpu.float() @ b_cpu.float()).half()
    result = kernel(a_cpu.cuda(), b_cpu.cuda()).cpu()
    torch.testing.assert_close(result, expected, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_ppu
def test_ppu_gemm_register_pipeline_preserves_clear_accum():
    @T.prim_func
    def main(A: T.Tensor((64, 128), T.float16), B: T.Tensor((128, 64), T.float16), C: T.Tensor((64, 64), T.float32)):
        with T.Kernel(1, threads=128):
            a = T.alloc_shared((64, 128), T.float16)
            b = T.alloc_shared((128, 64), T.float16)
            c = T.alloc_fragment((64, 64), T.float32)
            T.copy(A, a)
            T.copy(B, b)
            T.gemm(a, b, c, clear_accum=True)
            T.copy(c, C)

    kernel = tilelang.compile(main, out_idx=[2])
    source = kernel.get_kernel_source()
    for name in ("A_local_0", "A_local_1", "B_local_0", "B_local_1"):
        assert name in source
    import torch

    generator = torch.Generator().manual_seed(31)
    a = torch.randn((64, 128), generator=generator, dtype=torch.float16)
    b = torch.randn((128, 64), generator=generator, dtype=torch.float16)
    expected = a.float() @ b.float()
    for _ in range(3):
        torch.testing.assert_close(kernel(a.cuda(), b.cuda()).cpu(), expected, rtol=1e-2, atol=1e-2)
