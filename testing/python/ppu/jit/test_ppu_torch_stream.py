import pytest
import torch
import tilelang
import tilelang.testing
import tilelang.language as T
from tilelang import tvm


@tilelang.testing.requires_ppu_compute_version(1, 5)
@pytest.mark.parametrize("prior_ppu_stream", [False, True])
def test_torch_graph_uses_current_stream_and_restores_ppu_context(prior_ppu_stream):
    @T.prim_func
    def add_one(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "float32")):
        with T.Kernel(1, threads=128):
            for i in T.Parallel(256):
                B[i] = A[i] + 1

    kernel = tilelang.compile(add_one, target={"kind": "ppu", "arch": "ppu_15"}, execution_backend="tvm_ffi")
    device_id = torch.cuda.current_device()
    device = tvm.device(19, device_id)
    a = torch.full((256,), 41.0, device="cuda")
    b = torch.empty_like(a)
    capture_stream = torch.cuda.Stream()
    other_stream = torch.cuda.Stream()
    previous = other_stream.cuda_stream if prior_ppu_stream else 0
    with tvm.ffi.use_raw_stream(device, previous):
        capture_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(capture_stream):
            kernel(a, b)
        assert tvm.ffi.get_raw_stream(device) == previous
        torch.cuda.synchronize()
        graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(graph, stream=capture_stream):
            kernel(a, b)
        assert tvm.ffi.get_raw_stream(device) == previous
        torch.cuda.synchronize()
        a.fill_(17.0)
        b.fill_(float("nan"))
        graph.replay()
        torch.cuda.synchronize()
        torch.testing.assert_close(b, torch.full_like(b, 18.0))
        assert tvm.ffi.get_raw_stream(device) == previous
