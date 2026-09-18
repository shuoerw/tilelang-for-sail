"""PPU system-scope atomic operations."""

import ctypes
import os

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tvm.error import InternalError
from tvm.ir import Op


PPU_TARGET = "ppu"


def _system_atomic_kernel(dtype="int32"):
    @T.prim_func
    def kernel(data: T.Tensor((8,), dtype)):
        with T.Kernel(1, threads=32):
            if T.get_thread_binding() == 0:
                T.atomic_add(data[0], 1, memory_order="relaxed", scope="system")
                T.atomic_add(data[0], 1, memory_order="consume", scope="system")
                T.atomic_add(data[0], 1, memory_order="acquire", scope="system")
                T.atomic_add(data[0], 1, memory_order="release", scope="system")
                T.atomic_add(data[0], 1, memory_order="acq_rel", scope="system")
                T.atomic_add(data[0], 1, memory_order="seq_cst", scope="system")

                data[1] = T.atomic_load(data[0], memory_order="relaxed", scope="system")
                data[2] = T.atomic_load(data[0], memory_order="consume", scope="system")
                data[3] = T.atomic_load(data[0], memory_order="acquire", scope="system")
                data[4] = T.atomic_load(data[0], memory_order="seq_cst", scope="system")

                T.atomic_store(data[5], 11, memory_order="relaxed", scope="system")
                T.atomic_store(data[6], 12, memory_order="release", scope="system")
                T.atomic_store(data[7], 13, memory_order="seq_cst", scope="system")

    return kernel


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("dtype", ["int32", "uint32"])
def test_ppu_system_atomic_codegen(dtype):
    source = tilelang.lower(
        _system_atomic_kernel(dtype),
        target=PPU_TARGET,
        enable_device_compile=False,
    ).kernel_source
    assert source.count("PPUSystemAtomicAdd") == 6
    assert source.count("PPUSystemAtomicLoad") == 4
    assert source.count("PPUSystemAtomicStore") == 3


@tilelang.testing.requires_ppu
@pytest.mark.parametrize(("operation", "memory_order"), [("add", 6), ("load", 3), ("load", 4), ("store", 1), ("store", 2), ("store", 4)])
def test_ppu_system_atomic_raw_ir_rejects_invalid_order(operation, memory_order):
    # Bypass the Python API: the backend must reject malformed imported IR too.
    @T.prim_func
    def kernel(data: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=32):
            if T.get_thread_binding() == 0:
                if operation == "load":
                    data[0] = T.call_intrin("int32", Op.get("tl.atomic_load_elem_op"), T.access_ptr(data[0], "r"), memory_order, "system")
                else:
                    T.evaluate(
                        T.call_intrin(
                            "handle", Op.get(f"tl.atomic_{operation}_elem_op"), T.access_ptr(data[0], "rw"), 1, memory_order, "system"
                        )
                    )

    with pytest.raises(InternalError, match="received invalid memory_order"):
        tilelang.lower(kernel, target=PPU_TARGET, enable_device_compile=False)


@tilelang.testing.requires_ppu
@pytest.mark.parametrize(
    ("dtype", "torch_dtype"),
    [("int32", torch.int32), ("uint32", torch.uint32)],
)
def test_ppu_system_atomic_runtime(dtype, torch_dtype):
    compiled = tilelang.compile(
        _system_atomic_kernel(dtype),
        execution_backend="tvm_ffi",
        target=PPU_TARGET,
    )
    data = torch.zeros((8,), dtype=torch_dtype, device="cuda")
    compiled(data)
    torch.cuda.synchronize()
    torch.testing.assert_close(
        data,
        torch.tensor(
            [6, 6, 6, 6, 6, 11, 12, 13],
            dtype=torch_dtype,
            device="cuda",
        ),
    )


@tilelang.testing.requires_ppu
def test_ppu_system_atomic_vectorized_loop():
    @T.prim_func
    def kernel(data: T.Tensor((128,), T.int32)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            for i in T.vectorized(4):
                index = tx * 4 + i
                T.atomic_store(data[index], i, memory_order="release", scope="system")
                T.atomic_add(data[index], 1, scope="system")
                value = T.atomic_load(data[index], memory_order="acquire", scope="system")
                data[index] = value + 1

    compiled = tilelang.compile(kernel, execution_backend="tvm_ffi", target=PPU_TARGET)
    source = compiled.get_kernel_source()
    assert "PPUSystemAtomicAdd" in source
    assert "PPUSystemAtomicLoad" in source
    assert "PPUSystemAtomicStore" in source
    data = torch.zeros((128,), dtype=torch.int32, device="cuda")
    compiled(data)
    torch.cuda.synchronize()
    torch.testing.assert_close(data, torch.arange(2, 6, dtype=torch.int32, device="cuda").repeat(32))


@tilelang.testing.requires_ppu
def test_ppu_system_atomic_parallel_loop():
    @T.prim_func
    def kernel(data: T.Tensor((130,), T.int32), result: T.Tensor((130,), T.int32)):
        with T.Kernel(1, threads=32):
            for i in T.Parallel(130):
                T.atomic_store(data[i], i, memory_order="release", scope="system")
                T.atomic_add(data[i], 1, scope="system")
                result[i] = T.atomic_load(data[i], memory_order="acquire", scope="system")

    compiled = tilelang.compile(kernel, execution_backend="tvm_ffi", target=PPU_TARGET)
    data = torch.empty((130,), dtype=torch.int32, device="cuda")
    result = torch.empty_like(data)
    compiled(data, result)
    torch.cuda.synchronize()
    torch.testing.assert_close(result, torch.arange(1, 131, dtype=torch.int32, device="cuda"))


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("dtype", ["int32", "uint32"])
@pytest.mark.parametrize("memory_order", ["relaxed", "consume", "acquire", "release", "acq_rel", "seq_cst"])
def test_ppu_system_atomic_contended_add(dtype, memory_order):
    @T.prim_func
    def kernel(data: T.Tensor((1,), dtype)):
        with T.Kernel(32, threads=128):
            for _ in T.serial(8):
                T.atomic_add(data[0], 1, memory_order=memory_order, scope="system")

    compiled = tilelang.compile(kernel, execution_backend="tvm_ffi", target=PPU_TARGET)
    initial = -65536 if dtype == "int32" else 0xF0000000
    data = torch.full((1,), initial, dtype=getattr(torch, dtype), device="cuda")
    compiled(data)
    torch.cuda.synchronize()
    assert data.item() == initial + 32 * 128 * 8


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("dtype", ["int32", "uint32"])
def test_ppu_system_atomic_conditional_value(dtype):
    @T.prim_func
    def kernel(data: T.Tensor((32,), dtype), result: T.Tensor((32,), dtype)):
        with T.Kernel(1, threads=32):
            tx = T.get_thread_binding()
            T.atomic_store(data[tx], T.if_then_else(tx % 2 == 0, 11, 17), scope="system")
            T.atomic_add(data[tx], T.if_then_else(tx % 2 == 0, 3, 5), scope="system")
            result[tx] = T.atomic_load(data[tx], scope="system")

    compiled = tilelang.compile(kernel, execution_backend="tvm_ffi", target=PPU_TARGET)
    data = torch.empty((32,), dtype=getattr(torch, dtype), device="cuda")
    result = torch.empty_like(data)
    compiled(data, result)
    torch.cuda.synchronize()
    expected = torch.tensor([14, 22] * 16, dtype=getattr(torch, dtype), device="cuda")
    torch.testing.assert_close(result, expected)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        compiled(data, result)
    for _ in range(3):
        graph.replay()
    torch.cuda.synchronize()
    torch.testing.assert_close(result, expected)


@tilelang.testing.requires_ppu
@pytest.mark.parametrize(("store_order", "load_order"), [("release", "acquire"), ("release", "consume"), ("seq_cst", "seq_cst")])
def test_ppu_system_atomic_message_passing(store_order, load_order):
    # Two resident CTAs exchange ordinary payload writes through an atomic flag.
    # Bounded polling reports a failure instead of hanging the test device.
    @T.prim_func
    def kernel(payload: T.Tensor((16,), T.int32), flag: T.Tensor((1,), T.int32), errors: T.Tensor((2,), T.int32)):
        with T.Kernel(2, threads=32) as bx:
            if T.get_thread_binding() == 0:
                active = T.alloc_var(T.int32, init=1)
                failures = T.alloc_var(T.int32, init=0)
                for iteration in T.serial(128):
                    if active != 0:
                        polls = T.alloc_var(T.int32, init=0)
                        ready = T.alloc_var(T.int32, init=0)
                        while (ready == 0) & (polls < 100000):
                            ready = T.cast(T.atomic_load(flag[0], memory_order=load_order, scope="system") == bx, T.int32)
                            polls += 1
                        if ready != 0:
                            if bx == 0:
                                for j in T.serial(16):
                                    payload[j] = iteration * 16 + j + 1
                            else:
                                for j in T.serial(16):
                                    failures += T.cast(payload[j] != iteration * 16 + j + 1, T.int32)
                            T.atomic_store(flag[0], 1 - bx, memory_order=store_order, scope="system")
                        else:
                            failures += 1
                            active = 0
                errors[bx] = failures

    compiled = tilelang.compile(kernel, execution_backend="tvm_ffi", target=PPU_TARGET)
    payload = torch.zeros((16,), dtype=torch.int32, device="cuda")
    flag = torch.zeros((1,), dtype=torch.int32, device="cuda")
    errors = torch.empty((2,), dtype=torch.int32, device="cuda")
    for _ in range(10):
        flag.zero_()
        compiled(payload, flag, errors)
        torch.cuda.synchronize()
        assert errors.tolist() == [0, 0]


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("seq_cst", [False, True])
@pytest.mark.parametrize("use_add", [False, True])
def test_ppu_system_atomic_peer_message_passing(seq_cst, use_add):
    if torch.cuda.device_count() < 2 or not torch.cuda.can_device_access_peer(1, 0):
        pytest.skip("Requires two peer-accessible PPU devices")
    store_order = "seq_cst" if seq_cst else "release"
    load_order = "seq_cst" if seq_cst else "acquire"

    @T.prim_func
    def kernel(pointers: T.Tensor((1,), T.ptr), errors: T.Tensor((1,), T.int32), role: T.int32, rounds: T.int32):
        with T.Kernel(1, threads=32):
            shared = T.make_tensor(pointers[0], (128,), T.int32)
            if T.get_thread_binding() == 0:
                active = T.alloc_var(T.int32, init=1)
                failures = T.alloc_var(T.int32, init=0)
                for iteration in T.serial(rounds):
                    if active != 0:
                        polls = T.alloc_var(T.int32, init=0)
                        ready = T.alloc_var(T.int32, init=0)
                        while (ready == 0) & (polls < 1000000):
                            ready = T.cast(T.atomic_load(shared[0], memory_order=load_order, scope="system") == role, T.int32)
                            polls += 1
                        if ready != 0:
                            if role == 0:
                                for j in T.serial(16):
                                    shared[64 + j] = iteration * 16 + j + 1
                            else:
                                for j in T.serial(16):
                                    failures += T.cast(shared[64 + j] != iteration * 16 + j + 1, T.int32)
                            if use_add:
                                T.atomic_add(shared[0], T.if_then_else(role == 0, 1, -1), memory_order=store_order, scope="system")
                            else:
                                T.atomic_store(shared[0], 1 - role, memory_order=store_order, scope="system")
                        else:
                            failures += 1
                            active = 0
                errors[0] = failures

    runtime = ctypes.CDLL(os.path.join(os.environ["PPU_SDK"], "targets/x86_64-linux/lib/libhggcrt1.so"))
    runtime.hggcDeviceEnablePeerAccess.argtypes = [ctypes.c_int, ctypes.c_uint]
    runtime.hggcDeviceDisablePeerAccess.argtypes = [ctypes.c_int]
    runtime.hggcGetErrorName.argtypes = [ctypes.c_int]
    runtime.hggcGetErrorName.restype = ctypes.c_char_p
    with torch.cuda.device(1):
        torch.cuda.init()
        status = runtime.hggcDeviceEnablePeerAccess(0, 0)
        assert status == 0 or runtime.hggcGetErrorName(status) == b"hggcErrorPeerAccessAlreadyEnabled"
        runtime.hggcGetLastError()
    try:
        compiled = tilelang.compile(kernel, execution_backend="tvm_ffi", target=PPU_TARGET)
        shared = torch.zeros((128,), dtype=torch.int32, device="cuda:0")
        pointers = [torch.tensor([shared.data_ptr()], dtype=torch.int64, device=f"cuda:{i}") for i in range(2)]
        errors = [torch.empty((1,), dtype=torch.int32, device=f"cuda:{i}") for i in range(2)]
        # Load the module on both devices before starting the bounded handshake.
        for device in range(2):
            with torch.cuda.device(device):
                compiled(pointers[device], errors[device], device, 0)
                torch.cuda.synchronize()
        for _ in range(10):
            for device in range(2):
                with torch.cuda.device(device):
                    compiled(pointers[device], errors[device], device, 128)
            for device in range(2):
                torch.cuda.synchronize(device)
                assert errors[device].item() == 0
    finally:
        if status == 0:
            with torch.cuda.device(1):
                assert runtime.hggcDeviceDisablePeerAccess(0) == 0


def test_system_atomic_scope_validation():
    with pytest.raises(ValueError, match="Unsupported memory scope"):

        @T.prim_func
        def kernel(data: T.Tensor((1,), T.int32)):
            with T.Kernel(1, threads=1):
                T.atomic_load(data[0], scope="rank")


@pytest.mark.parametrize("memory_order", ["release", "acq_rel"])
def test_system_atomic_load_rejects_invalid_memory_order(memory_order):
    with pytest.raises(ValueError, match="system-scope atomic_load"):

        @T.prim_func
        def kernel(data: T.Tensor((1,), T.int32)):
            with T.Kernel(1, threads=1):
                data[0] = T.atomic_load(data[0], memory_order=memory_order, scope="system")


@pytest.mark.parametrize("memory_order", ["consume", "acquire", "acq_rel"])
def test_system_atomic_store_rejects_invalid_memory_order(memory_order):
    with pytest.raises(ValueError, match="system-scope atomic_store"):

        @T.prim_func
        def kernel(data: T.Tensor((1,), T.int32)):
            with T.Kernel(1, threads=1):
                T.atomic_store(data[0], 1, memory_order=memory_order, scope="system")


def test_system_atomic_rejects_unknown_memory_order():
    with pytest.raises(ValueError, match="memory_order='unordered'"):

        @T.prim_func
        def kernel(data: T.Tensor((1,), T.int32)):
            with T.Kernel(1, threads=1):
                T.atomic_add(data[0], 1, memory_order="unordered", scope="system")


def test_system_atomic_rejects_unsupported_dtype():
    with pytest.raises(TypeError, match="expected one of: int32, uint32"):

        @T.prim_func
        def kernel(data: T.Tensor((1,), T.float32)):
            with T.Kernel(1, threads=1):
                T.atomic_add(data[0], 1.0, scope="system")


if __name__ == "__main__":
    tilelang.testing.main()
