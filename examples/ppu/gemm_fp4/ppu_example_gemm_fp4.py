"""PPU FP4 (e2m1) GEMM example covering NN/NT/TN/TT layouts.

Uses PPU0015 mma.m16n16k64 with float4_e2m1fn operands and float32 accumulator.
"""

import torch

import tilelang
import tilelang.ppu.language as T
from tilelang.transform import PassConfigKey

# e2m1 magnitude LUT; bit 3 = sign.
_FP4_E2M1_MAG = (0.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0, 6.0)


def _float_to_e2m1_codes(x: torch.Tensor) -> torch.Tensor:
    """Quantize a float tensor to e2m1 codes (uint8 in [0, 15])."""
    x = x.clamp(-6.0, 6.0)
    sign = (x < 0).to(torch.uint8) << 3
    lut = torch.tensor(_FP4_E2M1_MAG, device=x.device, dtype=torch.float32)
    mag = (x.abs().unsqueeze(-1) - lut).abs().argmin(dim=-1).to(torch.uint8)
    return sign | mag


def _pack_fp4(codes: torch.Tensor) -> torch.Tensor:
    """Pack e2m1 codes (..., K) into int8 (..., K // 2); even idx -> low nibble."""
    lo = codes[..., 0::2]
    hi = codes[..., 1::2]
    return (lo | (hi << 4)).view(torch.int8)


def _dequant_fp4(packed: torch.Tensor) -> torch.Tensor:
    """int8 (..., K // 2) -> fp32 (..., K)."""
    u = packed.view(torch.uint8)
    lo = (u & 0x0F).to(torch.int64)
    hi = (u >> 4).to(torch.int64)
    codes = torch.stack([lo, hi], dim=-1).flatten(-2)
    mag_lut = torch.tensor(_FP4_E2M1_MAG, device=u.device, dtype=torch.float32)
    mag = mag_lut[codes & 0x7]
    sign = torch.where((codes & 0x8) != 0, -1.0, 1.0)
    return mag * sign


def _gen_fp4_inputs(A_shape, B_shape, device):
    """Quantize random fp16 to e2m1 on CPU, pack into int8, move to device."""
    a = torch.randn(*A_shape, dtype=torch.float16)
    b = torch.randn(*B_shape, dtype=torch.float16)
    a_packed = _pack_fp4(_float_to_e2m1_codes(a.float()))
    b_packed = _pack_fp4(_float_to_e2m1_codes(b.float()))
    return a_packed.to(device), b_packed.to(device)


def _make_matmul(pass_configs=None):
    @tilelang.jit(pass_configs=pass_configs)
    def matmul(A, B, block_M, block_N, block_K, num_stages, trans_A, trans_B, dtype=T.float4_e2m1fn, accum_dtype=T.float32):
        M, N, K = T.const("M, N, K")

        A: T.Tensor(((K, M) if trans_A else (M, K)), dtype)
        B: T.Tensor(((N, K) if trans_B else (K, N)), dtype)
        C = T.empty((M, N), accum_dtype)

        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
            A_shared = T.alloc_shared(((block_K, block_M) if trans_A else (block_M, block_K)), dtype)
            B_shared = T.alloc_shared(((block_N, block_K) if trans_B else (block_K, block_N)), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)

            T.copy(C_local, C[by * block_M, bx * block_N])

        return C

    return matmul


# AIU bulk copy and ldmat.swzl both enabled
matmul = _make_matmul(
    pass_configs={
        PassConfigKey.TL_DISABLE_AIU_LOWER: False,
        PassConfigKey.TL_DISABLE_LDMAT_SWZL: False,
    }
)

# Uses default pass_configs (TL_DISABLE_LDMAT_SWZL=False, swzl enabled on PPU15)
matmul_default = _make_matmul()


def run_case(M, N, K, block_M, block_N, block_K, num_stages, trans_A, trans_B, kernel_func=None):
    if kernel_func is None:
        kernel_func = matmul
    layout = f"{'T' if trans_A else 'N'}{'T' if trans_B else 'N'}"
    print(f"\n{'=' * 60}", flush=True)
    print(f"Testing layout={layout} num_stages={num_stages}", flush=True)
    print(f"{'=' * 60}", flush=True)

    compiled = kernel_func.compile(
        M=M, N=N, K=K, block_M=block_M, block_N=block_N, block_K=block_K, num_stages=num_stages, trans_A=trans_A, trans_B=trans_B
    )
    print("kernel compiled.", flush=True)

    device = torch.device("cuda")
    a_shape = (K, M) if trans_A else (M, K)
    b_shape = (N, K) if trans_B else (K, N)
    a_packed, b_packed = _gen_fp4_inputs(a_shape, b_shape, device)

    c = compiled(a_packed, b_packed)

    a_f32 = _dequant_fp4(a_packed.cpu())
    b_f32 = _dequant_fp4(b_packed.cpu())
    ref_c = (a_f32.T if trans_A else a_f32) @ (b_f32.T if trans_B else b_f32)

    torch.testing.assert_close(c.cpu(), ref_c, rtol=1e-2, atol=1e-2)
    print(f"All check passed. (layout={layout}, num_stages={num_stages})", flush=True)

    profiler = compiled.get_profiler()
    latency = profiler.do_bench(input_tensors=[a_packed, b_packed])
    print(f"tilelang Latency: {latency}ms", flush=True)


def main():
    M = N = K = 256
    block_M = block_N = block_K = 128

    for trans_A, trans_B in [
        (False, False),
        (True, False),
        (False, True),
        (True, True),
    ]:
        for num_stages in [0, 2]:
            run_case(M, N, K, block_M, block_N, block_K, num_stages, trans_A, trans_B)


if __name__ == "__main__":
    main()
