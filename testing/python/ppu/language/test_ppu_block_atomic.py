"""PPU block-scope atomic add/load/store on shared and global memory."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


PPU_TARGET = "ppu"


@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
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


@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
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


def _block_add_orders_kernel(dtype):
    @T.prim_func
    def kernel(data: T.Tensor((1,), dtype), previous: T.Tensor((6,), dtype)):
        with T.Kernel(1, threads=32):
            if T.get_thread_binding() == 0:
                previous[0] = T.atomic_add(data[0], 1, memory_order="relaxed", return_prev=True, scope="block")
                T.atomic_add(data[0], 1, memory_order="relaxed", scope="block")
                previous[1] = T.atomic_add(data[0], 1, memory_order="consume", return_prev=True, scope="block")
                T.atomic_add(data[0], 1, memory_order="consume", scope="block")
                previous[2] = T.atomic_add(data[0], 1, memory_order="acquire", return_prev=True, scope="block")
                T.atomic_add(data[0], 1, memory_order="acquire", scope="block")
                previous[3] = T.atomic_add(data[0], 1, memory_order="release", return_prev=True, scope="block")
                T.atomic_add(data[0], 1, memory_order="release", scope="block")
                previous[4] = T.atomic_add(data[0], 1, memory_order="acq_rel", return_prev=True, scope="block")
                T.atomic_add(data[0], 1, memory_order="acq_rel", scope="block")
                previous[5] = T.atomic_add(data[0], 1, memory_order="seq_cst", return_prev=True, scope="block")
                T.atomic_add(data[0], 1, memory_order="seq_cst", scope="block")

    return kernel


_BLOCK_ADD_DTYPES = ["int32", "uint32", "int64", "uint64", "float32", "float64"]


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("dtype", _BLOCK_ADD_DTYPES)
def test_block_atomic_add_codegen(dtype):
    source = tilelang.lower(_block_add_orders_kernel(dtype), target=PPU_TARGET, enable_device_compile=False).kernel_source
    assert source.count("PPUBlockAtomicAdd(") == 12


@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
@pytest.mark.parametrize("dtype", _BLOCK_ADD_DTYPES)
def test_block_atomic_add_orders_runtime(dtype):
    compiled = tilelang.compile(_block_add_orders_kernel(dtype), target=PPU_TARGET, execution_backend="tvm_ffi")
    data = torch.zeros(1, dtype=getattr(torch, dtype), device="cuda")
    previous = torch.empty(6, dtype=data.dtype, device="cuda")
    compiled(data, previous)
    torch.cuda.synchronize()
    assert data.item() == 12
    assert previous.cpu().tolist() == [0, 2, 4, 6, 8, 10]


@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
@pytest.mark.parametrize("dtype", _BLOCK_ADD_DTYPES)
@pytest.mark.parametrize("storage", ["shared", "global"])
@pytest.mark.parametrize("return_prev", [False, True])
def test_block_atomic_add_contended(dtype, storage, return_prev):
    initial = {"int32": -1024, "uint32": 0xF0000000, "int64": -(1 << 40), "uint64": 1 << 40}.get(dtype, 0)

    @T.prim_func
    def kernel(counters: T.Tensor((2,), dtype), result: T.Tensor((2,), dtype), tickets: T.Tensor((2, 512), dtype)):
        with T.Kernel(2, threads=128) as bx:
            tx = T.get_thread_binding()
            shared = T.alloc_shared((1,), dtype)
            counter = shared if storage == "shared" else counters
            index = 0 if storage == "shared" else bx
            if tx == 0:
                T.atomic_store(counter[index], T.cast(initial, dtype), memory_order="relaxed", scope="block")
            T.sync_threads()
            for iteration in T.serial(4):
                if return_prev:
                    tickets[bx, iteration * 128 + tx] = T.atomic_add(
                        counter[index], 1, memory_order="acq_rel", return_prev=True, scope="block"
                    )
                else:
                    T.atomic_add(counter[index], 1, memory_order="acq_rel", scope="block")
            T.sync_threads()
            if tx == 0:
                result[bx] = T.atomic_load(counter[index], memory_order="acquire", scope="block")

    compiled = tilelang.compile(kernel, target=PPU_TARGET, execution_backend="tvm_ffi")
    counters = torch.empty(2, dtype=getattr(torch, dtype), device="cuda")
    result = torch.empty_like(counters)
    tickets = torch.empty((2, 512), dtype=counters.dtype, device="cuda")
    for _ in range(4):
        compiled(counters, result, tickets)
        torch.cuda.synchronize()
        assert result.cpu().tolist() == [initial + 512] * 2
        if return_prev:
            for block_tickets in tickets.cpu().tolist():
                assert sorted(block_tickets) == list(range(initial, initial + 512))


@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
@pytest.mark.parametrize("storage", ["shared", "global"])
def test_block_atomic_add_acq_rel_message_passing(storage):
    @T.prim_func
    def kernel(state: T.Tensor((3,), "int32"), result: T.Tensor((2,), "int32")):
        with T.Kernel(1, threads=96):
            tx = T.get_thread_binding()
            shared = T.alloc_shared((3,), "int32")
            data = shared if storage == "shared" else state
            if tx == 0:
                data[0] = 0
                data[1] = 0
                data[2] = 0
            T.sync_threads()
            if tx == 0 or tx == 32:
                data[tx // 32] = (tx // 32 + 1) * 111
                previous = T.atomic_add(data[2], 1, memory_order="acq_rel", return_prev=True, scope="block")
                if previous == 1:
                    result[0] = data[0] + data[1]
            if tx == 64:
                ready = T.alloc_var("int32", init=0)
                attempts = T.alloc_var("int32", init=0)
                while ready != 2 and attempts < 100000:
                    ready = T.atomic_load(data[2], memory_order="acquire", scope="block")
                    attempts += 1
                result[1] = T.if_then_else(ready == 2, data[0] + data[1], -1)

    compiled = tilelang.compile(kernel, target=PPU_TARGET, execution_backend="tvm_ffi")
    state = torch.empty(3, dtype=torch.int32, device="cuda")
    result = torch.empty(2, dtype=torch.int32, device="cuda")
    for _ in range(32):
        result.fill_(-1)
        compiled(state, result)
        torch.cuda.synchronize()
        assert result.cpu().tolist() == [333, 333]


@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
@pytest.mark.parametrize("vectorized", [False, True])
def test_block_atomic_add_loop_runtime(vectorized):
    size = 128 if vectorized else 130

    @T.prim_func
    def kernel(data: T.Tensor((size,), "int32"), previous: T.Tensor((size,), "int32")):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            if vectorized:
                for i in T.vectorized(4):
                    previous[tx * 4 + i] = T.atomic_add(data[tx * 4 + i], 3, memory_order="acq_rel", return_prev=True, scope="block")
            else:
                for i in T.Parallel(size):
                    previous[i] = T.atomic_add(data[i], 3, memory_order="acq_rel", return_prev=True, scope="block")

    compiled = tilelang.compile(kernel, target=PPU_TARGET, execution_backend="tvm_ffi")
    data = torch.arange(size, dtype=torch.int32, device="cuda")
    previous = torch.empty_like(data)
    compiled(data, previous)
    torch.cuda.synchronize()
    torch.testing.assert_close(data, torch.arange(size, dtype=torch.int32, device="cuda") + 3)
    torch.testing.assert_close(previous, torch.arange(size, dtype=torch.int32, device="cuda"))


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("return_prev", [False, True])
@pytest.mark.parametrize(
    ("dtype", "order", "message"),
    [("float16", 0, "32/64-bit"), ("int32", 6, "invalid memory_order"), ("int32", -1, "invalid memory_order")],
)
def test_block_atomic_add_raw_ir_validation(dtype, order, message, return_prev):
    from tvm.error import InternalError
    from tvm.ir import Op

    @T.prim_func
    def kernel(data: T.Tensor((1,), dtype)):
        with T.Kernel(1, threads=32):
            if T.get_thread_binding() == 0:
                T.evaluate(
                    T.call_intrin(
                        dtype if return_prev else "handle",
                        Op.get("tl.atomic_add_ret_elem_op" if return_prev else "tl.atomic_add_elem_op"),
                        T.access_ptr(data[0], "rw"),
                        1,
                        order,
                        "block",
                    )
                )

    with pytest.raises(InternalError, match=message):
        tilelang.lower(kernel, target=PPU_TARGET, enable_device_compile=False)
