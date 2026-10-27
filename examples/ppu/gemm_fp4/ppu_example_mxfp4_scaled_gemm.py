"""PPU native MXFP4 GEMM with runtime packed E8M0 block scales."""

from pathlib import Path

import torch

import tilelang
import tilelang.ppu.language as T
from tilelang.env import ACTLIZE_INCLUDE_DIR
from tilelang.transform import PassConfigKey

from ppu_example_gemm_fp4 import _dequant_fp4, _gen_fp4_inputs


_NATIVE_FP4_MMA = "ppu.tc02.mma.mx.sync.aligned.m16n16k64.row.col.f32.f4.f4.f32"


def _pack_e8m0_pairs(low: torch.Tensor, high: torch.Tensor) -> torch.Tensor:
    """Pack the first/second block-32 E8M0 bytes into one uint16."""
    return (low.to(torch.int32) | (high.to(torch.int32) << 8)).to(torch.uint16)


def _dequant_mxfp4(values: torch.Tensor, scales: torch.Tensor) -> torch.Tensor:
    """Apply physical [K/64, rows] packed scale storage to FP4 rows."""
    packed = scales.to(torch.int32).T
    exponents = torch.stack((packed & 0xFF, (packed >> 8) & 0xFF), dim=-1).flatten(1)
    multipliers = torch.pow(2.0, exponents.to(torch.float32) - 127).repeat_interleave(32, dim=1)
    return _dequant_fp4(values) * multipliers


@tilelang.jit(
    pass_configs={
        PassConfigKey.TL_DISABLE_AIU_LOWER: False,
        PassConfigKey.TL_DISABLE_LDMAT_SWZL: False,
    }
)
def matmul_scaled(A, B, A_scale, B_scale, block_M, block_N, block_K, num_stages):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), T.float4_e2m1fn)
    B: T.Tensor((N, K), T.float4_e2m1fn)
    A_scale: T.Tensor((K // 64, M), T.uint16)
    B_scale: T.Tensor((K // 64, N), T.uint16)
    C = T.empty((M, N), T.bfloat16)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), T.float4_e2m1fn)
        B_shared = T.alloc_shared((block_N, block_K), T.float4_e2m1fn)
        if block_K == 64:
            A_scale_shared = T.alloc_shared((block_M,), T.uint16)
            B_scale_shared = T.alloc_shared((block_N,), T.uint16)
        else:
            A_scale_shared = T.alloc_shared((block_K // 64, block_M), T.uint16)
            B_scale_shared = T.alloc_shared((block_K // 64, block_N), T.uint16)
        C_local = T.alloc_fragment((block_M, block_N), T.float32)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[bx * block_N, k * block_K], B_shared)
            # Scale tensors use the benchmark ABI: physical [K/64, M/N].
            if block_K == 64:
                T.copy(A_scale[k, by * block_M], A_scale_shared)
                T.copy(B_scale[k, bx * block_N], B_scale_shared)
            else:
                scale_k = k * (block_K // 64)
                T.copy(A_scale[scale_k, by * block_M], A_scale_shared)
                T.copy(B_scale[scale_k, bx * block_N], B_scale_shared)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                T.bool(False),
                T.bool(True),
                scale_A=A_scale_shared,
                scale_B=B_scale_shared,
            )

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


def make_scale_cases(rows_a: int, rows_b: int, k_tiles: int):
    """Generate representative E8M0 block-scale configurations for testing."""
    one_a = torch.full((k_tiles, rows_a), 0x7F7F, dtype=torch.uint16)
    one_b = torch.full((k_tiles, rows_b), 0x7F7F, dtype=torch.uint16)

    # Deliberately make both operands and both K32 halves different.
    distinct_a = _pack_e8m0_pairs(
        torch.full((k_tiles, rows_a), 126),
        torch.full((k_tiles, rows_a), 129),
    )
    distinct_b = _pack_e8m0_pairs(
        torch.full((k_tiles, rows_b), 128),
        torch.full((k_tiles, rows_b), 125),
    )

    generator = torch.Generator().manual_seed(20260831)
    random_a = _pack_e8m0_pairs(
        torch.randint(124, 131, (k_tiles, rows_a), generator=generator),
        torch.randint(124, 131, (k_tiles, rows_a), generator=generator),
    )
    random_b = _pack_e8m0_pairs(
        torch.randint(124, 131, (k_tiles, rows_b), generator=generator),
        torch.randint(124, 131, (k_tiles, rows_b), generator=generator),
    )
    return (("scale_one", one_a, one_b), ("distinct_halves", distinct_a, distinct_b), ("random", random_a, random_b))


def check_kernel_source(num_stages, block_K=64):
    """Compile and verify the generated kernel source contains scaled-MMA intrinsics."""
    M = N = 128
    K = block_K
    kernel = matmul_scaled.compile(
        M=M,
        N=N,
        K=K,
        block_M=128,
        block_N=128,
        block_K=block_K,
        num_stages=num_stages,
    )
    source = kernel.get_kernel_source()
    assert "#include <tl_templates/ppu/instruction/mma.h>" in source
    assert "tl::mma_sync_scaled<" in source
    assert "tl::tix_ldmatrix_x1_trans(" in source
    actlize_header = Path(ACTLIZE_INCLUDE_DIR) / "cute/arch/mma_ppu0015.hpp"
    assert actlize_header.exists(), f"actlize header not found: {actlize_header}"
    assert _NATIVE_FP4_MMA in actlize_header.read_text(encoding="utf-8")
    print("Kernel Source:", flush=True)
    print(source, flush=True)


def run_scaled_case(num_stages, scale_name, a_scale, b_scale, a_packed, b_packed, M, N, K, block_K=64, block_M=128, block_N=128):
    """Run a single (num_stages, scale) MXFP4 scaled-GEMM case."""
    device = torch.device("cuda")
    kernel = matmul_scaled.compile(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        num_stages=num_stages,
    )
    result = kernel(a_packed, b_packed, a_scale.to(device), b_scale.to(device))
    ref_a = _dequant_mxfp4(a_packed.cpu(), a_scale)
    ref_b = _dequant_mxfp4(b_packed.cpu(), b_scale)
    reference = (ref_a @ ref_b.T).to(torch.bfloat16)
    torch.testing.assert_close(result.cpu(), reference, rtol=2e-2, atol=5e-1)
    print(f"All check passed. (num_stages={num_stages}, case={scale_name})", flush=True)


def run_warp_tile_case(M, N, block_M, block_N, K=64, block_K=64, num_stages=0, scale_name="random"):
    """Run an MXFP4 case whose warp tile is not the default 4x4 atoms.

    Covers scale-collector group splitting (warp tiles wider than four
    m16n16 atoms along M or N) and lane-group redirect (fewer than four
    atom groups along an operand dimension).
    """
    device = torch.device("cuda")
    a_packed, b_packed = _gen_fp4_inputs((M, K), (N, K), device)
    scale_cases = {name: (a_scale, b_scale) for name, a_scale, b_scale in make_scale_cases(M, N, K // 64)}
    a_scale, b_scale = scale_cases[scale_name]
    run_scaled_case(num_stages, scale_name, a_scale, b_scale, a_packed, b_packed, M, N, K, block_K, block_M, block_N)


def main():
    M = N = K = 128
    device = torch.device("cuda")

    a_packed, b_packed = _gen_fp4_inputs((M, K), (N, K), device)
    for num_stages in (0, 2):
        check_kernel_source(num_stages)
        for name, a_scale, b_scale in make_scale_cases(M, N, K // 64):
            run_scaled_case(num_stages, name, a_scale, b_scale, a_packed, b_packed, M, N, K)

    # Non-4x4-atom warp tiles: scale-collector group splitting (>4 atoms
    # along M or N) and lane-group redirect (<4 groups).
    for m, n, bm, bn in [
        (256, 64, 256, 64),  # warp tile 128x32: 8x2 atoms, split along M
        (64, 256, 64, 256),  # warp tile 32x128: 2x8 atoms, split along N
        (64, 64, 64, 64),  # warp tile 32x32: 2x2 atoms, redirect only
    ]:
        for num_stages in (0, 2):
            run_warp_tile_case(m, n, bm, bn, num_stages=num_stages)
    # Group splitting combined with a multi-atom K tile (2-D scale regions).
    run_warp_tile_case(256, 64, 256, 64, K=128, block_K=128, num_stages=2)


if __name__ == "__main__":
    main()
