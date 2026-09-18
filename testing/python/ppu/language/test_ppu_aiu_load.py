"""PPU AIU byte loads and shared-memory alignment."""

import re

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.ppu.transform
import tilelang.testing
from tvm import IRModule, tirx, transform
from tvm.error import InternalError
from tvm.ir import Op, assert_structural_equal
from tvm.target import Target


PPU_TARGET = "ppu"


def _linear_aiu_kernel():
    @T.prim_func
    def kernel(src: T.Tensor((640,), T.uint8), dst: T.Tensor((640,), T.uint8)):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((640,), T.uint8)
            T.ppu_aiu_load(
                T.access_ptr(shared[0], "w"),
                T.access_ptr(src[0], "r"),
                128,
                1,
                128,
                1,
                128,
                0,
                0,
                0,
                0,
            )
            T.ppu_aiu_load(
                T.access_ptr(shared[128], "w"),
                T.access_ptr(src[128], "r"),
                128,
                2,
                128,
                2,
                128,
                0,
                0,
                0,
                0,
            )
            T.ppu_aiu_load(
                T.access_ptr(shared[384], "w"),
                T.access_ptr(src[384], "r"),
                128,
                1,
                128,
                1,
                128,
                0,
                0,
                0,
                0,
            )
            T.ppu_aiu_load(
                T.access_ptr(shared[512], "w"),
                T.access_ptr(src[512], "r"),
                128,
                1,
                128,
                1,
                128,
                0,
                0,
                0,
                0,
            )
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            T.sync_threads()
            for i in T.Parallel(640):
                dst[i] = shared[i]

    return kernel


def _aiu_tile_kernel(cube_c=128, swzl_mode=0, is_swizzle=1, cube_w=1):
    @T.prim_func
    def kernel(src: T.Tensor((cube_w * cube_c,), T.uint8), dst: T.Tensor((cube_w * cube_c,), T.uint8)):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((cube_w * cube_c,), T.uint8)
            T.ppu_aiu_load(
                T.access_ptr(shared[0], "w"),
                T.access_ptr(src[0], "r"),
                cube_c,
                cube_w,
                cube_c,
                cube_w,
                cube_c,
                0,
                0,
                swzl_mode,
                is_swizzle,
            )
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            T.sync_threads()
            for i in T.Parallel(cube_w * cube_c):
                dst[i] = shared[i]

    return kernel


def _shared_aiu_kernel(scope):
    @T.prim_func
    def kernel(src: T.Tensor((128,), T.uint8), dst: T.Tensor((129,), T.uint8)):
        with T.Kernel(1, threads=32):
            pad = T.alloc_shared((16,), T.uint8, scope=scope)
            shared = T.alloc_shared((128,), T.uint8, scope=scope)
            tx = T.get_thread_binding()
            if tx < 16:
                pad[tx] = T.cast(tx, T.uint8)
            T.sync_threads()
            T.ppu_aiu_load(
                T.access_ptr(shared[0], "w"),
                T.access_ptr(src[0], "r"),
                128,
                1,
                128,
                1,
                128,
                0,
                0,
                0,
                0,
            )
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            T.sync_threads()
            for i in T.Parallel(128):
                dst[i] = shared[i]
            if tx == 0:
                dst[128] = pad[15]

    return kernel


@tilelang.testing.requires_ppu
def test_ppu_aiu_load_codegen():
    with Target(PPU_TARGET):
        source = tilelang.lower(
            _linear_aiu_kernel(),
            target=PPU_TARGET,
            enable_device_compile=False,
        ).kernel_source
    assert source.count("tl::aiu_load<false>") == 4
    assert source.count("tl::cp_async_commit()") == 1
    assert source.count("tl::cp_async_wait<0>()") == 1


@tilelang.testing.requires_ppu
def test_ppu_aiu_load_explicit_groups_are_not_reordered():
    src = tirx.Var("src", "handle")
    buffers = [tirx.decl_buffer((128,), "uint8", name=name, scope="shared") for name in ("A", "B", "C")]
    loads = [
        tirx.Evaluate(
            tirx.Call(
                "handle",
                Op.get("tl.ppu_aiu_load"),
                [buffer.access_ptr("w"), src, 128, 1, 128, 1, 128, 0, 0, 0],
                annotations={"no_implicit_async_commit_wait": tirx.IntImm("int32", 1)} if index == 1 else {},
            )
        )
        for index, buffer in enumerate(buffers)
    ]
    uses = [tirx.Evaluate(tirx.BufferLoad(buffer, [0])) for buffer in reversed(buffers)]
    body = tirx.SeqStmt(loads + uses)
    func = tirx.PrimFunc([src] + [buffer.data for buffer in buffers], body)
    result = tilelang.ppu.transform.ReorderAIULoads()(IRModule({"main": func}))
    assert_structural_equal(result["main"].body, body)


@tilelang.testing.requires_ppu
def test_ppu_aiu_load_implicit_sync():
    @T.prim_func
    def kernel(src: T.Tensor((128,), T.uint8), dst: T.Tensor((128,), T.uint8)):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((128,), T.uint8)
            # The compiler-generated form keeps implicit commit/wait handling.
            T.call_intrin(
                "handle",
                Op.get("tl.ppu_aiu_load"),
                T.access_ptr(shared[0], "w"),
                T.access_ptr(src[0], "r"),
                128,
                1,
                128,
                1,
                128,
                0,
                0,
                0,
            )
            for i in T.Parallel(128):
                dst[i] = shared[i]

    with Target(PPU_TARGET):
        source = tilelang.lower(kernel, target=PPU_TARGET, enable_device_compile=False).kernel_source
    assert source.count("tl::aiu_load(") == 1
    assert source.count("tl::cp_async_commit()") == 1
    assert source.count("tl::cp_async_wait<0>()") == 1
    compiled = tilelang.compile(kernel, execution_backend="tvm_ffi", target=PPU_TARGET)
    src = torch.arange(128, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    compiled(src, dst)
    torch.testing.assert_close(dst, src)


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("is_swizzle", [0, 1])
def test_ppu_aiu_load_strided_zero_padding(is_swizzle):
    @T.prim_func
    def kernel(src: T.Tensor((3, 256), T.uint8), dst: T.Tensor((512,), T.uint8)):
        with T.Kernel(1, threads=32):
            shared = T.alloc_shared((512,), T.uint8)
            T.ppu_aiu_load(
                T.access_ptr(shared[0], "w"),
                T.access_ptr(src[0, 0], "r"),
                192,
                3,
                128,
                4,
                T.uint64(256),
                128,
                1,
                0,
                is_swizzle,
            )
            T.ptx_commit_group()
            T.ptx_wait_group(0)
            T.sync_threads()
            for i in T.Parallel(512):
                dst[i] = shared[i]

    compiled = tilelang.compile(kernel, execution_backend="tvm_ffi", target=PPU_TARGET)
    src_cpu = torch.randint(256, (3, 256), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    dst = torch.empty((512,), dtype=torch.uint8, device="cuda")
    compiled(src_cpu.cuda(), dst)
    expected = torch.zeros((4, 128), dtype=torch.uint8)
    expected[:2, :64] = src_cpu[1:3, 128:192]
    indices = torch.arange(512)
    if is_swizzle:
        indices = indices ^ ((indices // 128) * 16)
    torch.testing.assert_close(dst.cpu(), expected.flatten()[indices], rtol=0, atol=0)


@tilelang.testing.requires_ppu
def test_ppu_aiu_load_warp_managed_group():
    @T.prim_func
    def kernel(src: T.Tensor((512,), T.uint8), dst: T.Tensor((2, 512), T.uint8)):
        with T.Kernel(2, threads=128) as bx:
            tx = T.get_thread_binding()
            shared = T.alloc_shared((512,), T.uint8)
            if tx < 32:
                for tile in T.serial(4):
                    if tile < 3 or bx == 0:
                        T.ppu_aiu_load(
                            T.access_ptr(shared[tile * 128], "w"),
                            T.access_ptr(src[0], "r"),
                            512,
                            1,
                            128,
                            1,
                            512,
                            tile * 128,
                            0,
                            0,
                            0,
                        )
                T.ptx_commit_group()
            T.sync_threads()
            if tx < 32:
                T.ptx_wait_group(0)
            T.sync_threads()
            for i in T.Parallel(512):
                if i < 384 or bx == 0:
                    dst[bx, i] = shared[i]

    with Target(PPU_TARGET):
        source = tilelang.lower(kernel, target=PPU_TARGET, enable_device_compile=False).kernel_source
    assert source.count("tl::cp_async_commit()") == 1
    assert source.count("tl::cp_async_wait<0>()") == 1
    compiled = tilelang.compile(kernel, execution_backend="tvm_ffi", target=PPU_TARGET)
    src = torch.arange(512, dtype=torch.int32, device="cuda").to(torch.uint8)
    dst = torch.full((2, 512), 255, dtype=torch.uint8, device="cuda")
    compiled(src, dst)
    torch.testing.assert_close(dst[0], src)
    torch.testing.assert_close(dst[1, :384], src[:384])
    assert torch.all(dst[1, 384:] == 255).item()


@tilelang.testing.requires_ppu
@pytest.mark.parametrize(
    ("cube_c", "swzl_mode", "is_swizzle", "message"),
    [(256, 0, 0, "cube_c must not exceed 128"), (64, 0, 1, "requires swzl_mode=1")],
)
def test_ppu_aiu_load_rejects_invalid_tile(cube_c, swzl_mode, is_swizzle, message):
    with Target(PPU_TARGET), pytest.raises(InternalError, match=message):
        tilelang.lower(
            _aiu_tile_kernel(cube_c, swzl_mode, is_swizzle),
            target=PPU_TARGET,
            enable_device_compile=False,
        )


@tilelang.testing.requires_ppu
@pytest.mark.parametrize(("scope", "merge_static"), [("shared.dyn", False), ("shared", False), ("shared", True)])
def test_ppu_aiu_load_shared_alignment(scope, merge_static):
    func = _shared_aiu_kernel(scope)
    config = {"tirx.merge_static_smem": merge_static}
    with Target(PPU_TARGET), transform.PassContext(config=config):
        source = tilelang.lower(
            func,
            target=PPU_TARGET,
            enable_device_compile=False,
        ).kernel_source
    if scope == "shared" and not merge_static:
        assert re.search(r"__shared__\s+__align__\(128\)\s+(?:uint8_t|uchar)\s+shared\[128\]", source)
        assert re.search(r"__shared__\s+__align__\(16\)\s+(?:uint8_t|uchar)\s+pad\[16\]", source)
    else:
        match = re.search(r"void\*\s+shared\s*=.*?buf_(?:dyn_)?shmem\s*\+\s*(\d+)", source)
        assert match is not None, source
        assert int(match.group(1)) % 128 == 0
        if merge_static:
            assert re.search(r"__shared__\s+__align__\(128\)\s+(?:uint8_t|uchar)\s+buf_shmem\[", source)
    compiled = tilelang.compile(func, execution_backend="tvm_ffi", target=PPU_TARGET, pass_configs=config)
    src = torch.arange(128, dtype=torch.uint8, device="cuda")
    dst = torch.empty((129,), dtype=torch.uint8, device="cuda")
    compiled(src, dst)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst[:128], src)
    assert dst[128].item() == 15


@tilelang.testing.requires_ppu
@pytest.mark.parametrize(("cube_c", "swzl_mode"), [(64, 1), (128, 0)])
@pytest.mark.parametrize("is_swizzle", [0, 1])
@pytest.mark.parametrize("cube_w", [1, 16])
def test_ppu_aiu_load_layout(cube_c, swzl_mode, is_swizzle, cube_w):
    compiled = tilelang.compile(
        _aiu_tile_kernel(cube_c, swzl_mode, is_swizzle, cube_w),
        execution_backend="tvm_ffi",
        target=PPU_TARGET,
    )
    src_cpu = torch.randint(256, (cube_w * cube_c,), dtype=torch.uint8, generator=torch.Generator().manual_seed(0))
    src = src_cpu.cuda()
    dst = torch.empty_like(src)
    compiled(src, dst)
    torch.cuda.synchronize()
    indices = torch.arange(cube_w * cube_c)
    if is_swizzle:
        # XOR 16-byte columns with row % 8 (128B), or (row % 8) // 2 (64B).
        # The permutation is self-inverse; inspect raw bytes, without unswizzling in the kernel.
        rows = indices // cube_c
        indices = indices ^ (((rows % 8) >> swzl_mode) * 16)
    torch.testing.assert_close(dst.cpu(), src_cpu[indices], rtol=0, atol=0)


@tilelang.testing.requires_ppu
def test_ppu_aiu_load_runtime():
    compiled = tilelang.compile(_linear_aiu_kernel(), execution_backend="tvm_ffi", target=PPU_TARGET)
    src = torch.arange(640, dtype=torch.uint8, device="cuda")
    dst = torch.empty_like(src)
    compiled(src, dst)
    torch.cuda.synchronize()
    torch.testing.assert_close(dst, src)


if __name__ == "__main__":
    tilelang.testing.main()
