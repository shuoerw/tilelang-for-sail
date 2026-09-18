"""PPU block-scope atomic load/store on shared and global memory."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


PPU_TARGET = "ppu"


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("dtype", ["int32", "uint32", "int64", "float32"])
def test_block_atomic_roundtrip(dtype):
    @T.prim_func
    def kernel(data: T.Tensor((128,), dtype), result: T.Tensor((128,), dtype)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            shared = T.alloc_shared((128,), dtype)
            for i in T.vectorized(4):
                index = tx * 4 + i
                T.atomic_store(shared[index], T.cast(index, dtype), memory_order="relaxed", scope="block")
                value = T.atomic_load(shared[index], memory_order="relaxed", scope="block")
                T.atomic_store(data[index], value, scope="block")
                result[index] = T.atomic_load(data[index], scope="block")

    compiled = tilelang.compile(kernel, target=PPU_TARGET, execution_backend="tvm_ffi")
    source = compiled.get_kernel_source()
    assert "AtomicLoad<hggc::thread_scope_block>" in source
    assert "AtomicStore<hggc::thread_scope_block>" in source
    data = torch.empty(128, dtype=getattr(torch, dtype), device="cuda")
    result = torch.empty_like(data)
    compiled(data, result)
    torch.cuda.synchronize()
    expected = torch.arange(128, device="cuda").to(data.dtype)
    torch.testing.assert_close(data, expected)
    torch.testing.assert_close(result, expected)


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("storage", ["shared", "global"])
@pytest.mark.parametrize(("store_order", "load_order"), [("release", "acquire"), ("release", "consume"), ("seq_cst", "seq_cst")])
def test_block_atomic_message_passing(storage, store_order, load_order):
    @T.prim_func
    def kernel(state: T.Tensor((2,), "int32"), result: T.Tensor((1,), "int32")):
        with T.Kernel(1, threads=64):
            tx = T.get_thread_binding()
            shared = T.alloc_shared((2,), "int32")
            data = shared if storage == "shared" else state
            if tx == 0:
                data[0] = 0
                data[1] = 0
            T.sync_threads()
            if tx == 0:
                data[0] = 12345
                T.atomic_store(data[1], 1, memory_order=store_order, scope="block")
            if tx == 32:
                ready = T.alloc_var("int32", init=0)
                attempts = T.alloc_var("int32", init=0)
                while ready == 0 and attempts < 100000:
                    ready = T.atomic_load(data[1], memory_order=load_order, scope="block")
                    attempts += 1
                result[0] = T.if_then_else(ready != 0, data[0], -1)

    compiled = tilelang.compile(kernel, target=PPU_TARGET, execution_backend="tvm_ffi")
    state = torch.empty(2, dtype=torch.int32, device="cuda")
    result = torch.empty(1, dtype=torch.int32, device="cuda")
    for _ in range(16):
        compiled(state, result)
        torch.cuda.synchronize()
        assert result.item() == 12345
