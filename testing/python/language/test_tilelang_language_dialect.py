import importlib

import pytest

import tilelang
import tilelang.language as T_default
import tilelang.language.common as T_comm
from tilelang.language.tir.exports import (
    CLASSIFIED_VENDOR_TIR_EXPORTS,
    CUDA_ONLY_TIR_EXPORTS,
    METAL_ONLY_TIR_EXPORTS,
    PPU_ONLY_TIR_EXPORTS,
    SHARED_LEGACY_TIR_EXPORTS,
)


CUDA_ONLY_NAMES = {
    "ClusterKernel",
    "CUDASourceCodeKernel",
    "pdl_trigger",
    "rng_init",
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
    "tvm_rdna_wmma",
}


def test_default_language_is_static_cuda_facade():
    from tilelang.cuda import language as cuda_language

    assert tilelang.language is T_default
    assert importlib.import_module("tilelang.language") is T_default
    assert T_default.__tilelang_dialect__ == "cuda"
    assert cuda_language.__tilelang_dialect__ == "cuda"
    assert set(T_default.__all__) == set(cuda_language.__all__)

    for name in T_default.__all__:
        assert getattr(T_default, name) is getattr(cuda_language, name)


def test_common_language_preserves_special_dsl_exports():
    assert T_comm.__tilelang_dialect__ == "common"
    assert "__log" in T_comm.__all__
    assert hasattr(T_comm, "__log")
    assert CUDA_ONLY_TIR_EXPORTS.isdisjoint(T_comm.__all__)
    assert METAL_ONLY_TIR_EXPORTS.isdisjoint(T_comm.__all__)


def test_cuda_language_composes_common_and_cuda_symbols():
    from tilelang.cuda import language as T
    from tilelang.cuda import debug as cuda_debug

    assert T.copy is T_comm.copy
    assert T.tcgen05_mma is T.tcgen05_gemm
    assert T.tcgen05_mma_blockscaled is T.tcgen05_gemm_blockscaled
    assert T.wgmma_mma is T.wgmma_gemm
    assert T.device_assert is cuda_debug.device_assert
    assert set(T.__all__) >= CUDA_ONLY_NAMES
    assert set(T.__all__) >= CUDA_ONLY_TIR_EXPORTS
    assert hasattr(T, "TCGEN05TensorCoreIntrinEmitter")
    assert hasattr(T, "WGMMATensorCoreIntrinEmitter")


def test_rocm_language_composes_common_and_rocm_symbols():
    from tilelang.rocm import language as T

    assert T.__tilelang_dialect__ == "rocm"
    assert T.copy is T_comm.copy
    assert T.mfma is T.tvm_mfma
    assert T.mfma_store is T.tvm_mfma_store
    assert set(T.__all__) >= ROCM_ONLY_NAMES
    assert hasattr(T, "MatrixCoreIntrinEmitter")
    assert hasattr(T, "make_mfma_swizzle_layout")
    assert set(T.__all__) >= SHARED_LEGACY_TIR_EXPORTS


@pytest.mark.parametrize(
    ("emitter_name", "method_name", "mcpu", "expected_call"),
    [
        ("MatrixCoreIntrinEmitter", "mfma", "gfx942", "T.tvm_mfma"),
        ("WMMAIntrinEmitter", "wmma", "gfx1100", "T.tvm_rdna_wmma"),
    ],
)
def test_rocm_emitters_resolve_backend_tir_intrinsics(emitter_name, method_name, mcpu, expected_call):
    from tilelang.rocm import intrinsics

    target = tilelang.tvm.target.Target({"kind": "hip", "mcpu": mcpu})
    emitter_cls = getattr(intrinsics, emitter_name)
    emitter = emitter_cls(
        a_dtype="float16",
        b_dtype="float16",
        accum_dtype="float32",
        block_row_warps=1,
        block_col_warps=1,
        warp_row_tiles=16,
        warp_col_tiles=16,
        chunk=16,
        target=target,
    )
    emit = getattr(emitter, method_name)
    a_size = emitter.warp_rows * emitter.local_size_a
    b_size = emitter.warp_cols * emitter.local_size_b
    c_size = emitter.warp_rows * emitter.warp_cols * emitter.local_size_out

    @T_comm.prim_func
    def main():
        a_local = T_comm.alloc_local((a_size,), "float16")
        b_local = T_comm.alloc_local((b_size,), "float16")
        c_local = T_comm.alloc_local((c_size,), "float32")
        emit(a_local, b_local, c_local)

    assert expected_call in main.script()


def test_metal_language_owns_simdgroup_tir_exports():
    from tilelang.metal import language as T

    assert set(T.__all__) >= METAL_ONLY_TIR_EXPORTS


def test_upstream_vendor_tir_exports_are_classified():
    import tvm.tirx.script.parser as upstream_tir_parser

    vendor_exports = {name for name in upstream_tir_parser.__all__ if name.startswith("ptx_") or "simdgroup" in name}
    assert vendor_exports <= CLASSIFIED_VENDOR_TIR_EXPORTS


@pytest.mark.parametrize(
    "module_name",
    [
        "tilelang.cpu.language",
        "tilelang.metal.language",
        "tilelang.rocm.language",
        "tilelang.webgpu.language",
    ],
)
def test_non_cuda_dialects_do_not_export_cuda_symbols(module_name):
    module = importlib.import_module(module_name)
    assert CUDA_ONLY_NAMES.isdisjoint(module.__all__)


@pytest.mark.parametrize(
    "module_name",
    [
        "tilelang.cpu.language",
        "tilelang.cuda.language",
        "tilelang.metal.language",
        "tilelang.webgpu.language",
    ],
)
def test_non_rocm_dialects_do_not_export_rocm_symbols(module_name):
    module = importlib.import_module(module_name)
    assert ROCM_ONLY_NAMES.isdisjoint(module.__all__)


def test_common_only_dialects_match_common_surface():
    from tilelang.cpu import language as cpu_language
    from tilelang.webgpu import language as webgpu_language

    assert set(cpu_language.__all__) == set(T_comm.__all__)
    assert set(webgpu_language.__all__) == set(T_comm.__all__)


def test_cuda_whole_module_implementations_live_under_cuda():
    from tilelang.cuda import language as T

    assert T.pdl_trigger.__module__ == "tilelang.cuda.language.pdl"
    assert T.cluster_sync.__module__ == "tilelang.cuda.language.cluster"
    assert T.rng_init.__module__ == "tilelang.cuda.language.random"
    assert T.ws.__module__ == "tilelang.cuda.language.warpgroup"


def test_ppu_language_composes_common_and_supported_intrinsics():
    from tilelang.ppu import language as T

    assert T.__tilelang_dialect__ == "ppu"
    assert set(T.__all__) >= set(T_comm.__all__)
    assert T.copy is T_comm.copy
    assert set(T.__all__) >= PPU_ONLY_TIR_EXPORTS
    assert {"ptx_ldmatrix", "ptx_mma", "ptx_mma_sp"} <= set(T.__all__)
    assert (CUDA_ONLY_NAMES - {"rng_init", "pdl_trigger"}).isdisjoint(T.__all__)
    assert ROCM_ONLY_NAMES.isdisjoint(T.__all__)
    assert METAL_ONLY_TIR_EXPORTS.isdisjoint(T.__all__)
    for name in (*PPU_ONLY_TIR_EXPORTS, "rng_init", "rng_rand", "rng_rand_float", "pdl_sync", "pdl_trigger"):
        assert getattr(T, name).__module__ == "tilelang.ppu.language"


@pytest.mark.parametrize(
    "module_name",
    [
        "tilelang.language.common",
        "tilelang.cpu.language",
        "tilelang.cuda.language",
        "tilelang.metal.language",
        "tilelang.rocm.language",
        "tilelang.webgpu.language",
    ],
)
def test_non_ppu_dialects_do_not_export_ppu_intrinsics(module_name):
    module = importlib.import_module(module_name)
    for name in PPU_ONLY_TIR_EXPORTS:
        assert name not in module.__all__
        assert not hasattr(module, name)


def test_ppu_intrinsics_construct_registered_ops():
    from tilelang.ppu import language as T
    from tvm import tirx

    pointer = tirx.Var("pointer", "handle")
    swizzle = T.tix_ldmatrix_swzl(False, 4, pointer, pointer, 0)
    assert swizzle.op.name == "tl.tix_ldmatrix_swzl"
    aiu = T.ppu_aiu_load(pointer, pointer, 32, 32, 16, 16, 64, 0, 0, 0)
    assert aiu.op.name == "tl.ppu_aiu_load"
    scaled = T.ptx_mma_scaled(
        "m16n16k32",
        "row",
        "col",
        "e2m1",
        "e2m1",
        "fp32",
        pointer,
        0,
        pointer,
        0,
        pointer,
        0,
        False,
        1,
        2,
        0,
        1,
        dtype="float32",
    )
    assert scaled.op.name == "tirx.ptx_mma"
    assert len(scaled.args) == 17
    assert T.rng_init(42, 0).op.name == "tl.rng_init"
    assert T.pdl_trigger().op.name == "tl.pdl_trigger"


def test_ppu_rng_rejects_cuda_generator():
    from tilelang.ppu import language as T

    with pytest.raises(ValueError, match="Unsupported PPU random generator"):
        T.rng_init(42, 0, generator="curandStatePhilox4_32_10_t")
