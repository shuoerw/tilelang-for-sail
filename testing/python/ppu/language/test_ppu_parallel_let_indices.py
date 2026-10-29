"""PPU port of testing/python/language/test_tilelang_language_parallel.py.

The CUDA original needs the CUDA backend compiled into the wheel; the PPU
copy below keeps the parallel-loop/let-index coverage available on PPU-only
builds.
"""

import pytest
import torch

import tilelang
import tilelang.ppu.language as T
import tilelang.testing

tilelang.testing.set_random_seed()

PPU_TARGET = "ppu"

PARALLEL_LET_INDICES_SHAPES = [(1, 2, 2, 0), (2, 8, 4, 0), (3, 5, 8, 0), (2, 8, 8, 4)]


@pytest.mark.parametrize("enable_let_inline", [False, True])
@pytest.mark.parametrize("rows,cols,threads,thread_start", PARALLEL_LET_INDICES_SHAPES)
@tilelang.testing.requires_ppu
def test_parallel_let_indices(rows, cols, threads, thread_start, enable_let_inline):
    size = 2 * rows * cols + 8

    @T.prim_func
    def main(A: T.Tensor((size,), T.int32), B: T.Tensor((size,), T.int32)):
        with T.Kernel(2, threads=threads) as bx:
            if T.get_thread_binding() >= thread_start:
                for i, j in T.Parallel(rows, cols):
                    offset = bx * rows * cols + i * cols
                    idx = offset + j + 4
                    B[idx] = A[idx] * 3 + idx
                for i, j in T.Parallel(rows, cols):
                    offset = bx * rows * cols + i * cols
                    idx = offset + j + 4
                    B[idx] = B[idx] + 5

    config = {tilelang.PassConfigKey.TL_SIMPLIFY: {tilelang.PassConfigKey.TL_SIMPLIFY_ENABLE_LET_INLINE: enable_let_inline}}
    kernel = tilelang.compile(main, target=PPU_TARGET, execution_backend="tvm_ffi", pass_configs=config)
    data = torch.arange(size, dtype=torch.int32, device="cuda")
    output = torch.full_like(data, -7)
    kernel(data, output)
    torch.cuda.synchronize()
    expected = torch.full_like(data, -7)
    expected[4:-4] = data[4:-4] * 4 + 5
    torch.testing.assert_close(output, expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
