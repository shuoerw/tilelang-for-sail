"""PPU codegen + runtime correctness test for elect_one_sync.

Verifies that:
1. Kernels using T.shuffle_elect(thread_extent) compile on PPU and emit
   the expected tl_shuffle_elect<N>() template instantiation (codegen).
2. The elected thread count matches the expected value at runtime, and
   elected positions are correct (runtime correctness).

This validates the cute::elect_one_sync() PPU polyfill added in
src/tl_templates/ppu/intrin.h.
"""

import pytest
import torch

import tilelang
import tilelang.ppu.language as T
import tilelang.testing

NUM_THREADS = 128


# ---------------------------------------------------------------------------
# Kernel factory (tilelang.jit style – used for runtime correctness)
# ---------------------------------------------------------------------------
@tilelang.jit(out_idx=[-1])
def _elect_kernel_jit(num_threads: int = NUM_THREADS, thread_extent: int = 0):
    """Each thread writes 1 (elected) or 0 (not elected) to Out[threadIdx]."""

    @T.prim_func
    def kernel(Out: T.Tensor((num_threads,), T.int32)):
        with T.Kernel(1, threads=num_threads):
            tx = T.get_thread_binding()
            elected = T.shuffle_elect(thread_extent)
            Out[tx] = elected

    return kernel


# ---------------------------------------------------------------------------
# Kernel factory (T.prim_func style – used for codegen assertion)
# ---------------------------------------------------------------------------
def _elect_kernel_codegen(thread_extent, num_threads=NUM_THREADS):
    """Build a minimal kernel that calls T.shuffle_elect(thread_extent)."""

    @T.prim_func
    def main(Out: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=num_threads):
            is_elected = T.shuffle_elect(thread_extent)
            if is_elected:
                Out[0] = 1

    return main


# ---------------------------------------------------------------------------
# Helper: compute expected election mask
# ---------------------------------------------------------------------------
def _expected_mask(num_threads: int, thread_extent: int) -> torch.Tensor:
    """Return a (num_threads,) int32 tensor with 1 at elected positions."""
    indices = torch.arange(num_threads, dtype=torch.int64)
    if thread_extent == 0:
        # Only thread 0 (warp0-lane0) is elected
        mask = indices == 0
    else:
        # First thread (lane0) of every group of `thread_extent` threads
        mask = (indices % thread_extent) == 0
    return mask.to(dtype=torch.int32)


# ---------------------------------------------------------------------------
# Test: codegen assertion (preserved from original)
# ---------------------------------------------------------------------------
@tilelang.testing.requires_ppu
@pytest.mark.parametrize("thread_extent", [0, 32, 128])
def test_elect_one_sync_codegen(thread_extent):
    """T.shuffle_elect() should compile on PPU and emit tl_shuffle_elect<N>()."""
    program = _elect_kernel_codegen(thread_extent)
    kernel = tilelang.compile(program, out_idx=[0])
    source = kernel.get_kernel_source()
    expected = f"tl_shuffle_elect<{thread_extent}>()"
    assert expected in source, f"Expected '{expected}' in generated PPU source.\nSource:\n{source}"


# ---------------------------------------------------------------------------
# Test: runtime correctness
# ---------------------------------------------------------------------------
@tilelang.testing.requires_ppu
@pytest.mark.parametrize(
    "thread_extent, expected_elected",
    [
        (0, 1),  # whole block → only warp0-lane0
        (32, NUM_THREADS // 32),  # per-warp → 4 elected
        (128, 1),  # whole block group → 1 elected
    ],
    ids=["extent_0_block", "extent_32_per_warp", "extent_128_full_block"],
)
def test_elect_one_sync_runtime(thread_extent, expected_elected):
    """Verify elected thread count and positions match expectations."""
    kernel = _elect_kernel_jit(NUM_THREADS, thread_extent)
    result = kernel()

    # --- elected count ---
    actual_elected = int(torch.sum(result).item())
    assert actual_elected == expected_elected, (
        f"thread_extent={thread_extent}: expected {expected_elected} elected threads, got {actual_elected}.\nResult tensor: {result}"
    )

    # --- elected positions ---
    ref = _expected_mask(NUM_THREADS, thread_extent).to(device=result.device)
    torch.testing.assert_close(
        result.cpu(),
        ref.cpu(),
        msg=lambda m: (
            f"thread_extent={thread_extent}: elected positions mismatch.\n"
            f"Got:      {result.cpu().tolist()}\n"
            f"Expected: {ref.cpu().tolist()}\n{m}"
        ),
    )


if __name__ == "__main__":
    tilelang.testing.main()
