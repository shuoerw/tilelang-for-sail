"""Runtime coverage for descending page copies, including an odd tail."""

import pytest
import torch

import tilelang
import tilelang.language as T
from tilelang import tvm

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or tvm.get_global_func("target.build.tilelang_ppu", allow_missing=True) is None,
    reason="Requires a PPU device and PPU codegen",
)


@pytest.mark.parametrize("stages", [0, 2])
@pytest.mark.parametrize("pages", [4, 8])
def test_reverse_paired_page_copy(stages, pages):
    blocks = pages * 4

    @T.prim_func
    def main(A: T.Tensor((pages, 32), "int32"), B: T.Tensor((blocks, 32), "int32")):
        with T.Kernel(blocks, threads=32) as bx:
            shared = T.alloc_shared((2, 32), "int32")
            acc = T.alloc_fragment((32,), "int32")
            T.clear(acc)
            last_page = (pages * 2 - 1 - bx // 2) // 2
            for pair in T.Pipelined(T.ceildiv(last_page + 1, 2), num_stages=stages):
                for slot in T.unroll(2, explicit=True):
                    page = last_page - pair * 2 - slot
                    if page >= 0:
                        T.copy(A[page, :], shared[slot, :], prefer_instruction="cp_async")
                for slot in T.unroll(2, explicit=True):
                    page = last_page - pair * 2 - slot
                    if page >= 0:
                        for i in T.Parallel(32):
                            acc[i] += shared[slot, i]
            T.copy(acc, B[bx, :])

    kernel = tilelang.compile(main, target="ppu", pass_configs={"tl.disable_aiu_lower": True})
    for seed in (0, 17):
        torch.manual_seed(seed)
        src = torch.randint(-32, 33, (pages, 32), dtype=torch.int32, device="cuda")
        original = src.clone()
        expected = torch.stack([src[: pages - bx // 4].sum(dim=0) for bx in range(blocks)])
        dst = torch.empty((blocks, 32), dtype=torch.int32, device="cuda")
        for _ in range(5):
            kernel(src, dst)
            torch.testing.assert_close(dst, expected.to(torch.int32), rtol=0, atol=0)
            torch.testing.assert_close(src, original, rtol=0, atol=0)
