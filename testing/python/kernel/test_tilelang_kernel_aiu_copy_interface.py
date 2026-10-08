"""Functional tests for the PPU AIU copy preference."""

import torch

import tilelang
import tilelang.ppu.language as T
import tilelang.testing
from tilelang.transform import PassConfigKey


M = 128
N = 128
K = 64


def _kernel(copy_mode):
    @T.prim_func
    def main(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((N, K), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(1, 1, threads=128) as (bx, by):
            A_shared = T.alloc_shared((128, 64), T.float16)
            B_shared = T.alloc_shared((128, 64), T.float16)
            C_local = T.alloc_fragment((128, 128), T.float32)

            T.clear(C_local)
            if copy_mode == "plain":
                T.copy(A[0, 0], A_shared)
                T.copy(B[0, 0], B_shared)
            elif copy_mode == "preferred":
                T.copy(A[0, 0], A_shared, prefer_instruction="aiu")
                T.copy(B[0, 0], B_shared, prefer_instruction="aiu")
            else:
                raise ValueError(copy_mode)

            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
            T.copy(C_local, C)

    return main


def _compile(copy_mode, enabled=True):
    configs = {PassConfigKey.TL_DISABLE_AIU_LOWER: not enabled}
    return tilelang.compile(_kernel(copy_mode), out_idx=[-1], pass_configs=configs)


def _source(kernel):
    return kernel.get_kernel_source()


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_forced_aiu_copy_api_is_removed():
    assert not hasattr(T, "aiu_copy")


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_plain_copy_does_not_use_aiu():
    source = _source(_compile("plain"))
    assert "tl::aiu_load" not in source


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_preferred_copy_uses_aiu_and_gets_sync():
    source = _source(_compile("preferred"))
    assert "tl::aiu_load" in source
    assert "tl::cp_async_commit()" in source
    assert "tl::cp_async_wait" in source


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_preferred_aiu_copy_can_be_disabled():
    source = _source(_compile("preferred", enabled=False))
    assert "tl::aiu_load" not in source


@tilelang.testing.requires_ppu_compute_version(1, 5)
def test_stage0_aiu_fallback_cp_async_keeps_sync_semantics():
    @T.prim_func
    def main(
        A: T.Tensor((13, 512), T.float16),
        B: T.Tensor((13, 512), T.float16),
    ):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((13, 512), T.float16)
            T.copy(A, shared, prefer_instruction="aiu")
            for i, j in T.Parallel(13, 512):
                B[i, j] = shared[i, j]

    kernel = tilelang.compile(main, out_idx=[-1], pass_configs={PassConfigKey.TL_DISABLE_AIU_LOWER: False})
    a = torch.randn(13, 512, device="cuda", dtype=torch.float16)
    torch.testing.assert_close(kernel(a), a)

    source = _source(kernel)
    assert "tl::aiu_load" not in source
    assert "tl::cp_async" in source
    assert "tl::cp_async_commit()" in source
    assert "tl::cp_async_wait<0>" in source
