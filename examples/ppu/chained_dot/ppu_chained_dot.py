import tilelang
import tilelang.ppu.language as T
from tilelang.transform import PassConfigKey
import torch


def print_float16_hex(tensor):
    assert tensor.dtype == torch.float16
    uint_tensor = tensor.view(torch.uint16)
    # 转为 numpy 或 list 保持形状
    hex_array = [[f"0x{v:04X}" for v in row] for row in uint_tensor.tolist()]
    for row in hex_array:
        print(row)


# Chained dot product: a @ b @ b @ b (i.e., a * b^3).
# Pure sequential chain—each result feeds the next multiplication with the same matrix b.
@tilelang.jit(
    out_idx=[3, 4, 5], verbose=False, pass_configs={PassConfigKey.TL_DISABLE_AIU_LOWER: False, PassConfigKey.TL_DISABLE_LDMAT_SWZL: False}
)
def matmul(M, N, K, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    thread_num = min((M // block_M) * (N // block_N) * 32, 128)

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        V: T.Tensor((N, N), dtype),
        C: T.Tensor((M, N), dtype),
        D: T.Tensor((M, N), dtype),
        E: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=thread_num) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            V_shared = T.alloc_shared((block_N, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            C_local_cast = T.alloc_fragment((block_M, block_N), dtype)
            D_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            D_local_cast = T.alloc_fragment((block_M, block_N), dtype)
            E_local = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_local)
            T.clear(D_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=1):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.copy(V[k * block_N, bx * block_N], V_shared)
                T.gemm(A_shared, B_shared, C_local)
                T.copy(C_local, C_local_cast)
                T.gemm(C_local_cast, B_shared, D_local)
                T.copy(D_local, D_local_cast)
                T.gemm(D_local_cast, B_shared, E_local)

            T.copy(C_local, C[by * block_M, bx * block_N])
            T.copy(D_local, D[by * block_M, bx * block_N])
            T.copy(E_local, E[by * block_M, bx * block_N])

    return gemm


def main():
    torch.set_printoptions(linewidth=480, sci_mode=False)

    shape = 64
    block_size = 64

    a = torch.randn(shape, shape).cuda().half()
    b = torch.randn(shape, shape).cuda().half()
    v = torch.randn(shape, shape).cuda().half()

    print("a:")
    print_float16_hex(a)
    print("b:")
    print_float16_hex(b)
    print("v:")
    print_float16_hex(v)

    ref_c = a @ b
    ref_d = ref_c @ b
    ref_e = ref_d @ b

    print("ref_c:")
    print_float16_hex(ref_c)
    print(ref_c)
    print("ref_d:")
    print_float16_hex(ref_d)
    print(ref_d)
    print("ref_e:")
    print_float16_hex(ref_e)
    print(ref_e)

    # c = torch.zeros(ref_c.size(), dtype=torch.float16, device=ref_c.device)
    # d = torch.zeros(ref_d.size(), dtype=torch.float16, device=ref_d.device)
    kernel = matmul(shape, shape, shape, block_size, block_size, block_size)
    c, d, e = kernel(a, b, v)

    print("c:")
    print_float16_hex(c)
    print(c)
    print("d:")
    print_float16_hex(d)
    print(d)
    print("e:")
    print_float16_hex(e)
    print(e)

    # Get CUDA Source
    print("CUDA Source:")
    print(kernel.get_kernel_source())

    torch.testing.assert_close(c, ref_c)
    torch.testing.assert_close(d, ref_d)
    torch.testing.assert_close(e, ref_e)
    print("All check passed.")


if __name__ == "__main__":
    main()
