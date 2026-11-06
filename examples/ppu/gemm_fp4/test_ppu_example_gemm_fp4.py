"""Combined tests for PPU FP4 GEMM and MXFP4 scaled GEMM examples.

Parametrized to give independent pytest cases per layout / num_stages / scale,
replacing the two monolithic test files with a single, fine-grained suite.
"""

import pytest

import tilelang.testing
import ppu_example_gemm_fp4
import ppu_example_mxfp4_scaled_gemm


# ---------------------------------------------------------------------------
# FP4 (e2m1) GEMM – 4 layouts × 2 pipeline depths = 8 cases
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "trans_A,trans_B",
    [
        (False, False),  # NN
        (False, True),  # TN
        (True, False),  # NT
        (True, True),  # TT
    ],
)
@pytest.mark.parametrize("num_stages", [0, 2])
@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_gemm_fp4(trans_A, trans_B, num_stages):
    M = N = K = 256
    ppu_example_gemm_fp4.run_case(
        M,
        N,
        K,
        block_M=128,
        block_N=128,
        block_K=64,
        num_stages=num_stages,
        trans_A=trans_A,
        trans_B=trans_B,
    )


# ---------------------------------------------------------------------------
# FP4 (e2m1) GEMM – block_K=128 with/without swzl pass_config
# ---------------------------------------------------------------------------


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_gemm_fp4_block_k128_with_swzl():
    """block_K=128 with swzl pass_config: fp4 128*4/8=64B -> swzl_mode=1."""
    M = N = K = 256
    ppu_example_gemm_fp4.run_case(
        M,
        N,
        K,
        block_M=128,
        block_N=128,
        block_K=128,
        num_stages=2,
        trans_A=False,
        trans_B=True,
        kernel_func=ppu_example_gemm_fp4.matmul,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_gemm_fp4_block_k128_without_swzl():
    """block_K=128, swzl disabled: fp4 ldmatrix emitted as plain tix_ldmatrix_x4."""
    M = N = K = 256
    ppu_example_gemm_fp4.run_case(
        M,
        N,
        K,
        block_M=128,
        block_N=128,
        block_K=128,
        num_stages=2,
        trans_A=False,
        trans_B=True,
        kernel_func=ppu_example_gemm_fp4.matmul_default,
    )


# ---------------------------------------------------------------------------
# MXFP4 scaled GEMM – kernel source sanity check (2 stages x 3 K tiles = 6 cases)
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("block_K", [64, 128, 192])
@pytest.mark.parametrize("num_stages", [0, 2])
@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_mxfp4_scaled_gemm_kernel_source(num_stages, block_K):
    ppu_example_mxfp4_scaled_gemm.check_kernel_source(num_stages, block_K)


# ---------------------------------------------------------------------------
# MXFP4 scaled GEMM – functional correctness (2 stages x 3 K tiles x 3 scale cases = 18)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def mxfp4_inputs():
    """Generate packed FP4 operands once per module; skipped on non-PPU hosts."""
    import torch

    M = N = 128
    return {
        K: (
            *ppu_example_mxfp4_scaled_gemm._gen_fp4_inputs(
                (M, K),
                (N, K),
                torch.device("cuda"),
            ),
            M,
            N,
            K,
        )
        for K in (64, 128, 192)
    }


@pytest.mark.parametrize("num_stages", [0, 2])
@pytest.mark.parametrize("block_K", [64, 128, 192])
@pytest.mark.parametrize("scale_name", ["scale_one", "distinct_halves", "random"])
@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_mxfp4_scaled_gemm(num_stages, block_K, scale_name, mxfp4_inputs):
    a_packed, b_packed, M, N, K = mxfp4_inputs[block_K]
    scale_cases = {name: (a_scale, b_scale) for name, a_scale, b_scale in ppu_example_mxfp4_scaled_gemm.make_scale_cases(M, N, K // 64)}
    a_scale, b_scale = scale_cases[scale_name]
    ppu_example_mxfp4_scaled_gemm.run_scaled_case(
        num_stages,
        scale_name,
        a_scale,
        b_scale,
        a_packed,
        b_packed,
        M,
        N,
        K,
        block_K,
    )


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_mxfp4_scaled_gemm_rejects_subatom_k():
    with pytest.raises(AssertionError, match="one packed uint16 pair per K=64 MMA atom"):
        ppu_example_mxfp4_scaled_gemm.matmul_scaled.compile(
            M=128,
            N=128,
            K=32,
            block_M=128,
            block_N=128,
            block_K=32,
            num_stages=0,
        )


# ---------------------------------------------------------------------------
# MXFP4 scaled GEMM – non-4x4-atom warp tiles (3 shapes x 2 stages = 6 cases)
# Covers scale-collector group splitting (>4 atoms along M/N) and the
# lane-group redirect path (<4 groups along an operand dimension).
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "M,N,block_M,block_N",
    [
        (256, 64, 256, 64),  # warp tile 128x32: 8x2 atoms, split along M
        (64, 256, 64, 256),  # warp tile 32x128: 2x8 atoms, split along N
        (64, 64, 64, 64),  # warp tile 32x32: 2x2 atoms, redirect only
        (192, 64, 192, 64),  # warp tile 96x32: 6x2 atoms, A partial chunk
        (64, 192, 64, 192),  # warp tile 32x96: 2x6 atoms, B partial chunk
    ],
)
@pytest.mark.parametrize("num_stages", [0, 2])
@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_mxfp4_scaled_gemm_warp_tiles(M, N, block_M, block_N, num_stages):
    ppu_example_mxfp4_scaled_gemm.run_warp_tile_case(M, N, block_M, block_N, K=64, block_K=64, num_stages=num_stages)


@tilelang.testing.requires_ppu
@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
def test_mxfp4_scaled_gemm_warp_split_multi_k():
    """Group splitting combined with a multi-atom K tile (2-D scale regions)."""
    ppu_example_mxfp4_scaled_gemm.run_warp_tile_case(256, 64, 256, 64, K=128, block_K=128, num_stages=2)


if __name__ == "__main__":
    tilelang.testing.main()
