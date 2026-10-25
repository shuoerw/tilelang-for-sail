"""Cooperative grid barriers: visibility, reuse, and launch isolation."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


PPU_TARGET = {"kind": "ppu", "arch": "ppu_15"}


def _visibility_kernel(grid, block, iterations=32):
    gx, gy, gz = grid
    tx, ty, tz = block
    blocks = gx * gy * gz
    threads = tx * ty * tz
    elements = blocks * threads

    @T.prim_func
    def kernel(
        data: T.Tensor((elements,), T.int32),
        errors: T.Tensor((elements,), T.int32),
        seed: T.int32,
    ):
        with T.Kernel(gx, gy, gz, threads=block) as (bx, by, bz):
            thread = T.get_thread_binding(0) + tx * (T.get_thread_binding(1) + ty * T.get_thread_binding(2))
            block_id = bx + gx * (by + gy * bz)
            index = block_id * threads + thread
            peer = ((block_id + 1) % blocks) * threads + threads - 1 - thread
            failures = T.alloc_local((1,), T.int32)
            failures[0] = 0
            for iteration in T.serial(iterations):
                data[index] = seed + iteration * elements + index
                T.sync_grid()
                failures[0] += T.if_then_else(data[peer] != seed + iteration * elements + peer, 1, 0)
                # Finish all reads before any block starts the next write.
                T.sync_grid()
            # Accumulate across launches so later success cannot hide a failure.
            errors[index] += failures[0]

    return kernel


def _compile(grid, block, **kwargs):
    return tilelang.compile(
        _visibility_kernel(grid, block),
        target=PPU_TARGET,
        execution_backend="tvm_ffi",
        **kwargs,
    )


def _buffers(grid, block):
    elements = grid[0] * grid[1] * grid[2] * block[0] * block[1] * block[2]
    return (
        torch.empty(elements, dtype=torch.int32, device="cuda"),
        torch.zeros(elements, dtype=torch.int32, device="cuda"),
    )


def _check(data, errors, seed):
    torch.testing.assert_close(errors, torch.zeros_like(errors), rtol=0, atol=0)
    expected = torch.arange(data.numel(), dtype=torch.int32, device=data.device)
    expected += seed + 31 * data.numel()
    torch.testing.assert_close(data, expected, rtol=0, atol=0)


def test_ppu_grid_sync_codegen():
    artifact = tilelang.lower(
        _visibility_kernel((2, 1, 1), (32, 1, 1)),
        target=PPU_TARGET,
        enable_device_compile=False,
    )
    source = artifact.kernel_source
    assert "#include <tl_templates/ppu/grid_sync.h>" in source
    assert source.count("tl::sync_grid();") >= 2
    assert "cooperative_groups::this_grid().sync();" not in source
    assert any(int(func.attrs.get("use_cooperative_groups", 0)) == 1 for func in artifact.device_mod.functions.values())


def test_ppu_no_grid_sync_codegen():
    @T.prim_func
    def kernel(output: T.Tensor((32,), T.int32)):
        with T.Kernel(1, threads=32):
            output[T.get_thread_binding()] = 0

    artifact = tilelang.lower(kernel, target=PPU_TARGET, enable_device_compile=False)
    assert "tl_templates/ppu/grid_sync.h" not in artifact.kernel_source


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("compile_flags", [[], ["-D_HCG_ABI_VER=2"]])
def test_ppu_grid_sync_compile(compile_flags):
    # Compile both the optimized path and the SDK fallback without a GPU launch.
    compiled = _compile((3, 1, 1), (128, 1, 1), compile_flags=compile_flags)
    assert "tl::sync_grid();" in compiled.get_kernel_source()


@tilelang.testing.requires_ppu
@pytest.mark.parametrize("grid", [(1, 1, 1), (3, 1, 1), (39, 1, 1), (2, 2, 2)])
@pytest.mark.parametrize("block", [(32, 1, 1), (128, 1, 1), (512, 1, 1), (16, 2, 2)])
def test_ppu_grid_sync_visibility(grid, block):
    compiled = _compile(grid, block)
    data, errors = _buffers(grid, block)
    for seed in (7, 101, 2003):
        compiled(data, errors, seed)
    _check(data, errors, 2003)


@tilelang.testing.requires_ppu
def test_ppu_grid_sync_streams():
    grid, block = (3, 1, 1), (128, 1, 1)
    compiled = _compile(grid, block)
    streams = [torch.cuda.Stream() for _ in range(2)]
    buffers = [_buffers(grid, block) for _ in streams]
    # Compile and initialize before launching from independent streams.
    for data, errors in buffers:
        compiled(data, errors, 0)
    torch.cuda.synchronize()
    for iteration in range(16):
        for stream_id, stream in enumerate(streams):
            with torch.cuda.stream(stream):
                compiled(*buffers[stream_id], 1000 * stream_id + iteration)
    for stream_id, stream in enumerate(streams):
        stream.synchronize()
        _check(*buffers[stream_id], 1000 * stream_id + 15)


@tilelang.testing.requires_ppu
def test_ppu_grid_sync_graph_replay():
    grid, block = (3, 1, 1), (128, 1, 1)
    compiled = _compile(grid, block)
    capture_stream = torch.cuda.Stream()
    buffers = [_buffers(grid, block) for _ in range(2)]
    graphs = [torch.cuda.CUDAGraph() for _ in buffers]
    for data, errors in buffers:
        compiled(data, errors, 0)
    torch.cuda.synchronize()
    # Separate captures on the same stream must not share mutable barrier state
    # when their graph executables are replayed from different streams.
    for graph_id, graph in enumerate(graphs):
        with torch.cuda.graph(graph, stream=capture_stream):
            compiled(*buffers[graph_id], 1000 * graph_id)
            compiled(*buffers[graph_id], 1000 * graph_id + 1)
    streams = [torch.cuda.Stream() for _ in graphs]
    for _ in range(16):
        for stream, graph in zip(streams, graphs):
            with torch.cuda.stream(stream):
                graph.replay()
    for graph_id, stream in enumerate(streams):
        stream.synchronize()
        _check(*buffers[graph_id], 1000 * graph_id + 1)


@tilelang.testing.requires_ppu
def test_ppu_grid_sync_sdk_fallback():
    grid, block = (3, 1, 1), (128, 1, 1)
    # Select another SDK ABI namespace to exercise the guarded SDK fallback.
    compiled = _compile(grid, block, compile_flags=["-D_HCG_ABI_VER=2"])
    data, errors = _buffers(grid, block)
    compiled(data, errors, 23)
    _check(data, errors, 23)


if __name__ == "__main__":
    tilelang.testing.main()
