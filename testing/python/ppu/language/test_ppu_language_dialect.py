"""PPU-runnable copy of testing/python/language/test_tilelang_language_dialect.py.

The shared-suite copy still asserts the pre-refactor identity
``T.copy is T_comm.copy``; the v0.1.15 refactor moved the typed copy hints
into the dialects, so the PPU dialect owns a ``copy`` shadow (consumed by
test_ppu_aiu_load). This copy asserts the post-refactor contract; the
original case stays skip-listed on PPU boards.
"""

from tilelang.language import common as T_comm
from tilelang.language.tir.exports import (
    METAL_ONLY_TIR_EXPORTS,
    PPU_ONLY_TIR_EXPORTS,
)

CUDA_ONLY_NAMES = {
    "ClusterKernel",
    "CUDASourceCodeKernel",
    "mma_gemm_blockscaled",
    "pdl_trigger",
    "rng_init",
    "tcgen05_gemm_blockscaled",
    "tcgen05_mma",
    "tma_copy",
    "wgmma_mma",
    "ws",
}

ROCM_ONLY_NAMES = {
    "MatrixCoreIntrinEmitter",
    "WMMAIntrinEmitter",
    "ds_read_tr16_b64",
    "ds_read_tr8_b64",
    "mfma",
    "rdna_wmma",
    "tvm_mfma",
    "tvm_mfma_store",
    "tvm_rdna_wmma",
    "tvm_rdna_wmma_store",
}


def test_ppu_language_composes_common_and_supported_intrinsics():
    from tilelang.ppu import language as T

    assert T.__tilelang_dialect__ == "ppu"
    assert set(T.__all__) >= set(T_comm.__all__)
    # The PPU dialect shadows T.copy with typed backend hints
    # (prefer_instruction); see the CUDA dialect test for the contract.
    assert T.copy is not T_comm.copy
    assert set(T.__all__) >= PPU_ONLY_TIR_EXPORTS
    assert {"ptx_ldmatrix", "ptx_mma", "ptx_mma_sp"} <= set(T.__all__)
    assert (CUDA_ONLY_NAMES - {"rng_init", "pdl_trigger"}).isdisjoint(T.__all__)
    assert ROCM_ONLY_NAMES.isdisjoint(T.__all__)
    assert METAL_ONLY_TIR_EXPORTS.isdisjoint(T.__all__)
    for name in (*PPU_ONLY_TIR_EXPORTS, "rng_init", "rng_rand", "rng_rand_float", "pdl_sync", "pdl_trigger"):
        assert getattr(T, name).__module__ == "tilelang.ppu.language"
