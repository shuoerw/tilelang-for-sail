from __future__ import annotations

from tilelang.backend.execution_backend import ExecutionBackendSpec

PPU_EXECUTION_BACKENDS = [
    ExecutionBackendSpec(
        "tvm_ffi",
        enable_host_codegen=True,
        enable_device_compile=True,
    ),
    ExecutionBackendSpec("cython"),
]
