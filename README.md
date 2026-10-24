# TileLang for PPU

## Introduction

TileLang is a concise, efficient DSL built on top of TVM for developing high-performance GPU/CPU kernels with Pythonic syntax. This repository extends TileLang with a complete PPU hardware backend, enabling it to compile and generate kernels that run natively on PPU hardware.

Based on TileLang v0.1.12 (upstream commit `2d63708c`). For the full upstream documentation — DSL syntax, programming guides, tutorials, and benchmarks — see [README_UPSTREAM.md](./README_UPSTREAM.md) or the [TileLang GitHub](https://github.com/tile-ai/tilelang).

## PPU Backend Extensions and Optimizations

The PPU backend adds native code generation, runtime integration, and hardware-specific optimizations for PPU architectures (`ppu0010` / `ppu0015`).

- **PPU Python module (`tilelang.ppu`)**: The entry point to PPU-specific functionality, exposing hand-written MMA intrinsic emitters and swizzle layout helpers (`tilelang.ppu.intrinsics`), PPU operator registration (`tilelang.ppu.op`), PPU-specific transforms (`tilelang.ppu.transform`), and the PPU pass pipeline (`tilelang.ppu.pipeline`).
- **PPU-native code generation**: A dedicated codegen backend (`codegen_ppu`) that translates TileLang IR into PPU hardware instructions, with PPU-specific intrinsic rules, layout inference, and uniform variable injection, independent of CUDA SM versioning.
- **AIU asynchronous data movement**: Asynchronous Instruction Unit (AIU) support for latency hiding, including AIU load reordering optimization, sync barrier injection, and swizzle-aligned warp slicing for efficient data prefetching.
- **Multi-precision MMA dispatch**: Hardware-specific MMA instruction dispatchers covering FP32, FP16, BF16, FP8 (E4M3FN/E5M2), MXFP4 scaled, and INT8 precisions, with per-architecture atom shape selection for `ppu0010` and `ppu0015`.
- **GEMM layout management**: PPU-specific GEMM layout functions handling all transpose combinations (NN/NT/TN/TT), chained RS vs normal RS differentiation, and `ldmatrix.swzl` modes for sub-16-bit data types.

## PPU Pass Configs

Two PPU-specific pass configs control hardware optimization features. AIU lowering is **disabled by default** and must be explicitly enabled via `pass_configs`; `ldmatrix.swzl` is **enabled by default** on `ppu0015`:

| Config Key | Description |
|------------|-------------|
| `tl.disable_aiu_lower` | When set to `False`, enables AIU (Asynchronous Instruction Unit) lowering, which replaces global→shared memory copies with asynchronous AIU load instructions for latency hiding. |
| `tl.disable_ldmat_swzl` | Defaults to `False`, so `ldmatrix.swzl` mode is enabled on `ppu0015`: data is loaded from swizzled shared memory directly into MMA fragments, reducing shared memory bank conflicts. Set to `True` to fall back to plain `ldmatrix` in `T.gemm` and `T.copy`. |

Example usage:

```python
from tilelang.transform import PassConfigKey

pass_configs = {
    PassConfigKey.TL_DISABLE_AIU_LOWER: False,      # Enable AIU lowering
    PassConfigKey.TL_DISABLE_LDMAT_SWZL: False,     # Default; set True to disable ldmatrix.swzl
}

@tilelang.jit(pass_configs=pass_configs)
def my_kernel(...):
    ...
```

See [examples/ppu/gemm/ppu_example_gemm.py](./examples/ppu/gemm/ppu_example_gemm.py) for a complete example with pass configs.

## Requirements

- PPU SDK
- ZW 810 / 810E (`ppu0010`) or ZW M890 (`ppu0015`) hardware
- Python >= 3.10
- PyTorch 2.0 or later (a Docker image matching the PPU SDK version is recommended)

## Installation

For system dependencies and detailed build options, refer to [README_UPSTREAM.md](./README_UPSTREAM.md).

The PPU backend is enabled by default (`-DUSE_PPU=ON` in `pyproject.toml`).

## Quick Start

The following example defines, compiles, runs, and verifies an FP16 GEMM kernel with FP32 accumulation. It uses PyTorch CUDA tensors; PyTorch uses the same `cuda` device name on PPU systems. TileLang selects the target automatically from the current environment.

```python
import tilelang
import tilelang.language as T

# @tilelang.jit(target="cuda")
# target can be "auto", a bare kind such as "cuda", or a config dict such as
# {"kind": "cuda", "arch": "sm_90"}.
@tilelang.jit
def matmul(A, B, block_M=128, block_N=128, block_K=32,
           dtype=T.float16, accum_dtype=T.float32):
    M, N, K = T.const("M, N, K")

    A: T.Tensor((M, K), dtype)
    B: T.Tensor((K, N), dtype)
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local  = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)

        T.copy(C_local, C[by * block_M, bx * block_N])

    return C

import torch
a = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
b = torch.randn(1024, 1024, device="cuda", dtype=torch.float16)
c = matmul(a, b)
torch.testing.assert_close(c, a @ b, rtol=1e-2, atol=1e-2)
```

See [examples/ppu/](./examples/ppu/) for more examples covering GEMM variants, attention, hand-written MMA intrinsics, and other operators.

## Testing

```bash
# Run PPU unit tests
pytest testing/python/ppu/ -v

# Run PPU example tests
pytest examples/ppu/ -v
```

## Repository Structure

```
src/ppu/
├── codegen/            # PPU code generation & intrinsics
├── layout/             # PPU GEMM layout functions
├── op/                 # PPU operator lowering
├── runtime/            # PPU runtime & device API
├── target/             # PPU target definition
└── transform/          # PPU-specific compiler passes

src/tl_templates/ppu/   # PPU hardware template headers
tilelang/ppu/           # Python-level PPU backend
examples/ppu/           # Ready-to-run PPU examples
testing/python/ppu/     # PPU unit tests
```

## Acknowledgement

TileLang for PPU is built upon the [TileLang](https://github.com/tile-ai/tilelang) project by Tile-AI and the [TVM](https://github.com/apache/tvm) compiler infrastructure.
