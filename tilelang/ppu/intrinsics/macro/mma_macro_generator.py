from __future__ import annotations
import tilelang.language as T
from typing import Literal
from collections.abc import Callable
from tilelang.transform import PassConfigKey, get_pass_context
from tvm import DataType
from tvm import tirx
from tvm.ir import Range
from tvm.tirx import PrimExpr, IndexMap, Buffer, Var, BufferRegion, BufferLoad
from tilelang import tvm as tvm
from tvm.runtime import convert
from ..layout.utils import (
    mma_store_index_map,
    mma_store_index_map_fp64,
    get_ldmatrix_offset,
)
from tilelang.utils import is_fragment, get_buffer_region_from_load
from ..layout.mma_layout import (
    # Standard layout functions
    shared_16x8_to_mma_32x4_layout_sr_a,
    shared_16x8_to_mma_32x4_layout_sr_b,
    shared_16x16_to_mma_32x8_layout_sr_a,
    shared_16x16_to_mma_32x8_layout_sr_b,
    shared_16x32_to_mma_32x16_layout_sr_a,
    shared_16x32_to_mma_32x16_layout_sr_b,
    mma_load_a_32x4_to_shared_16x8_layout,
    mma_load_b_32x4_to_shared_16x8_layout,
    mma_load_b_32x8_to_shared_16x16_layout,
    mma_load_a_32x16_to_shared_16x32_layout,
    mma_load_b_32x16_to_shared_16x32_layout,
    mma_load_a_32x32_to_shared_16x64_layout,
    mma_load_b_32x32_to_shared_16x64_layout,
    mma_load_a_32x8_to_shared_16x16_layout,
    ldmatrix_32x8_to_shared_16x16_layout,
    ldmatrix_32x16_to_shared_16x32_layout_a,
    ldmatrix_32x16_to_shared_16x32_layout_b,
    ldmatrix_32x16_to_shared_16x64_layout_a,
    ldmatrix_32x16_to_shared_16x64_layout_b,
    mma_store_32x8_to_shared_16x16_layout,
    # PPU-specific layout functions
    ppu_shared_16x16_to_mma_32x8_layout_trans_sr_b,
    ppu_ldmatrix_32x4_to_shared_16x8_layout_a,
    ppu_ldmatrix_32x8_to_shared_16x16_layout,
    ppu_ldmatrix_trans_32x8_to_shared_16x16_layout,
    ppu_ldmatrix_32x16_to_shared_16x32_layout_s8_a,
    ppu_mma_store_32x8_to_shared_16x16_layout,
    ppu_mma_load_a_32x4_to_shared_16x8_layout,
    ppu_shared_16x8_to_mma_32x4_layout_sr_a,
    ppu_shared_16x16_to_mma_32x8_layout_sr_a,
    ppu_shared_16x16_to_mma_32x8_layout_trans_sr_a,
    ppu_shared_16x16_to_mma_32x8_layout_sr_a_from_gemm_c,
)

lift = convert


class TensorCoreIntrinEmitter:
    """
    To eliminate Python syntax within TIR Macro.
    """

    M_DIM = 16
    # use lowercase as n_dim can be dynamic
    # the smallest instructions can be m16n8k16, so the n_dim can also be 8
    n_dim = 16
    WARP_SIZE = 32
    dtype_abbrv = {
        "float16": "fp16",
        "bfloat16": "bf16",
        "float32": "fp32",
        "float64": "fp64",
        "int4": "int4",
        "int8": "int8",
        "uint8": "uint8",
        "int32": "int32",
        "float8_e4m3": "e4m3",
        "float8_e4m3fn": "e4m3",
        "float8_e4m3fnuz": "e4m3",
        "float8_e5m2": "e5m2",
        "float8_e5m2fnuz": "e5m2",
        "float6_e2m3fn": "e2m3",
        "float6_e3m2fn": "e3m2",
        "float4_e2m1fn": "e2m1",
        "custom[float4_e2m1_unpacked]8": "e2m1",
        "custom[tfloat32]": "tf32",
    }

    # Represent the thread binding in the form of (tx, warp_n, warp_m)
    is_m_first: bool = False
    warp_rows: int = 1
    warp_cols: int = 1

    def __init__(
        self,
        a_dtype: str = T.float16,
        b_dtype: str = T.float16,
        accum_dtype: str = T.float16,
        a_transposed: bool = False,
        b_transposed: bool = False,
        block_row_warps: int = 2,
        block_col_warps: int = 2,
        warp_row_tiles: int = 8,
        warp_col_tiles: int = 8,
        chunk: int = 16,
        reduce_k: int = 1,
        num_elems_per_byte: int = 1,
        is_m_first: bool | None = False,
        thread_var: Var | None = None,
    ):
        self.a_dtype = a_dtype
        self.b_dtype = b_dtype
        self.accum_dtype = accum_dtype
        self.a_transposed = a_transposed
        self.b_transposed = b_transposed
        # Hint Information
        self.block_row_warps = block_row_warps
        self.block_col_warps = block_col_warps
        self.warp_row_tiles = warp_row_tiles
        self.warp_col_tiles = warp_col_tiles
        self.chunk = chunk
        self._initialize_k_dim(self.a_dtype)
        self._initialize_m_dim(self.a_dtype)
        self._initialize_micro_size(self.M_DIM, self.k_dim)
        self._initialize_local_size(self.M_DIM, self.n_dim, self.k_dim, self.WARP_SIZE)
        self._initialize_abbrev(self.a_dtype, self.b_dtype, accum_dtype)
        self._initialize_mma_prefix(self.k_dim)
        self._initialize_is_m_first(is_m_first)

        self.reduce_k = reduce_k
        self.threads = self.WARP_SIZE * (block_row_warps * block_col_warps) * reduce_k
        self.num_elems_per_byte = num_elems_per_byte
        self.thread_var = thread_var

        if self.warp_rows == 0 or self.warp_cols == 0:
            raise ValueError(
                f"Invalid threads configuration for this tile shape, {self.warp_rows} x {self.warp_cols} with threads {self.threads}"
            )

    def _initialize_k_dim(self, a_dtype=T.float16):
        if isinstance(a_dtype, str):
            a_dtype = DataType(a_dtype)
        self.k_dim = min(256 // a_dtype.bits, self.chunk)

    def _initialize_m_dim(self, a_dtype=T.float16):
        if isinstance(a_dtype, str):
            a_dtype = DataType(a_dtype)
        if a_dtype.bits == 64:
            # FP64 MMA uses m8n8k4; n_dim is set by _initialize_micro_size.
            self.M_DIM = 8

    def _initialize_local_size(self, m_dim=16, n_dim=16, k_dim=16, warp_size=32):
        self.local_size_a = (m_dim * k_dim) // warp_size
        self.local_size_b = (n_dim * k_dim) // warp_size
        self.local_size_out = (m_dim * n_dim) // warp_size

    def _initialize_abbrev(self, a_dtype, b_dtype, accum_dtype):
        self.a_dtype_abbrv = self._get_dtype_abbrv(a_dtype)
        self.b_dtype_abbrv = self._get_dtype_abbrv(b_dtype)
        self.accum_dtype_abbrv = self._get_dtype_abbrv(accum_dtype)
        if self._should_use_tf32_mma_operand(a_dtype, accum_dtype):
            self.a_dtype_abbrv = "tf32"
        if self._should_use_tf32_mma_operand(b_dtype, accum_dtype):
            self.b_dtype_abbrv = "tf32"

    def _get_dtype_abbrv(self, dtype: str) -> str:
        if "float4_e2m1_unpacked" in dtype:
            return "e2m1"
        if dtype not in self.dtype_abbrv:
            raise ValueError(f"Unsupported dtype: {dtype}")
        return self.dtype_abbrv[dtype]

    @staticmethod
    def _should_use_tf32_mma_operand(dtype: str, accum_dtype: str) -> bool:
        operand_dtype = DataType(dtype)
        accumulator_dtype = DataType(accum_dtype)
        return str(operand_dtype) == "float32" and str(accumulator_dtype) == "float32"

    def _initialize_mma_prefix(self, k_dim: int = 16):
        if k_dim == 4:
            # fp64
            self.mma_prefix = "m8n8k4"
        elif k_dim == 8:
            # typically used for tfloat32
            self.mma_prefix = "m16n8k8"
        elif k_dim == 16:
            # typically used for float16/bfloat16
            self.mma_prefix = "m16n8k16"
        elif k_dim == 32:
            # typically used for int8/fp8
            # sometimes int4/uint4 is also supported
            self.mma_prefix = "m16n8k32"
        elif k_dim == 64:
            # typically used for int4/uint4
            self.mma_prefix = "m16n8k64"
        elif k_dim == 128:
            # typically used for int2/uint2
            self.mma_prefix = "m16n8k128"
        elif k_dim == 256:
            # typically used for uint1
            self.mma_prefix = "m16n8k256"
        else:
            raise ValueError(f"Unsupported k_dim {k_dim}")

    def _initialize_micro_size(self, m_dim: int = 16, k_dim: int = 16):
        warp_row_tiles = self.warp_row_tiles
        warp_col_tiles = self.warp_col_tiles
        if k_dim == 4:
            assert m_dim == 8, f"For fp64 MMA, m_dim must be 8, got {m_dim}"
            self.n_dim = 8
            self.micro_size_y = 8
            self.warp_rows = warp_row_tiles // m_dim
            self.warp_cols = warp_col_tiles // 8
        else:
            assert warp_row_tiles >= 16, f"warp_row_tiles must be greater than 16, got {warp_row_tiles}"
            assert warp_row_tiles % 16 == 0, f"warp_row_tiles must be divisible by 16, got {warp_row_tiles}"
            assert warp_col_tiles >= 8, f"warp_col_tiles must be greater than 8, got {warp_col_tiles}"
            assert warp_col_tiles % 8 == 0, f"warp_col_tiles must be divisible by 8, got {warp_col_tiles}"

            self.warp_rows = warp_row_tiles // m_dim

            if warp_col_tiles % 16 == 0:
                self.n_dim = 16
                self.micro_size_y = 16
                self.warp_cols = warp_col_tiles // 16
            else:
                # must be divisible by 8
                self.n_dim = 8
                self.micro_size_y = 8
                self.warp_cols = warp_col_tiles // 8

        self.micro_size_x = m_dim
        self.micro_size_k = k_dim

    def _initialize_is_m_first(self, is_m_first: bool | None = False):
        if is_m_first is not None:
            self.is_m_first = is_m_first

    def get_thread_binding(self):
        if self.thread_var is None:
            current_frame = T.KernelLaunchFrame.Current()
            assert current_frame is not None, "Must be called in a T.Kernel Frame"
            return current_frame.get_thread_binding()
        else:
            return self.thread_var

    def _use_fp64_store_index_map(self) -> bool:
        # m8n8 MMA atoms produce two C registers and share the FP64 lane map.
        return DataType(self.accum_dtype).bits == 64 or self.local_size_out == 2

    def get_store_index_map(self, inverse: bool = False) -> IndexMap:
        warp_size, local_size_c = self.WARP_SIZE, self.local_size_out
        if self._use_fp64_store_index_map():
            index_map = IndexMap.from_func(mma_store_index_map_fp64, index_dtype=T.int32)
        else:
            index_map = IndexMap.from_func(mma_store_index_map, index_dtype=T.int32)
        if not inverse:
            return index_map
        inverse_index_map = index_map.inverse([warp_size, local_size_c])
        return inverse_index_map

    def extract_thread_binding(self, thread_id: PrimExpr, is_m_first: bool | None = None) -> tuple[PrimExpr, PrimExpr, PrimExpr]:
        """
        is_m_first: True if the thread binding is in the form of (tx, warp_n, warp_m)
        which represents [warp_size, block_row_warps (split n), block_col_warps (split m)]
        Otherwise, it is in the form of [warp_size, block_col_warps (split m), block_row_warps (split n)]
        """
        WARP_SIZE = self.WARP_SIZE
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps

        # if is_m_first is None, then use the default value
        if is_m_first is None:
            is_m_first = self.is_m_first

        if is_m_first:
            lane_id, warp_n, warp_m = (
                thread_id % WARP_SIZE,
                (thread_id // WARP_SIZE) % block_col_warps,
                (thread_id // (WARP_SIZE * block_col_warps)) % block_row_warps,
            )
            return lane_id, warp_n, warp_m
        else:
            lane_id, warp_m, warp_n = (
                thread_id % WARP_SIZE,
                (thread_id // WARP_SIZE) % block_row_warps,
                (thread_id // (WARP_SIZE * block_row_warps)) % block_col_warps,
            )
            return lane_id, warp_n, warp_m

    def ldmatrix_a(self, A_local_buf: Buffer, A_shared_buf: Buffer | BufferRegion, ki: PrimExpr, rk: PrimExpr | None = 0):
        # Fast path for fp64: no ldmatrix support, do direct per-lane loads
        a_dtype = self.a_dtype
        if DataType(a_dtype).bits == 64:
            warp_row_tiles = self.warp_row_tiles
            warp_rows = self.warp_rows
            chunk = self.chunk
            micro_size_x = self.micro_size_x  # 8
            micro_size_k = self.micro_size_k  # 4
            local_size_a = self.local_size_a  # 1
            a_transposed = self.a_transposed

            thread_binding = self.get_thread_binding()
            # legalize shared buffer to region
            A_region = self._legalize_to_buffer_region(A_shared_buf)
            A_buf = A_region.buffer
            A_base0 = A_region.region[-2].min
            A_base1 = A_region.region[-1].min
            A_other = [r.min for r in A_region.region[:-2]]

            @T.macro
            def _warp_ld_a_fp64(
                A_local_buf,
                A_shared_buf,
                ki,
                thread_binding,
                rk=0,
            ):
                tx, _, warp_m = self.extract_thread_binding(thread_binding)
                for i in T.serial(warp_rows):
                    wi = warp_m * warp_row_tiles + i * micro_size_x
                    wk = rk * chunk + ki * micro_size_k
                    mi = tx // micro_size_k
                    mk = tx % micro_size_k
                    if a_transposed:
                        A_local_buf[i * local_size_a] = A_buf[tuple(A_other) + (A_base0 + wk + mk, A_base1 + wi + mi)]
                    else:
                        A_local_buf[i * local_size_a] = A_buf[tuple(A_other) + (A_base0 + wi + mi, A_base1 + wk + mk)]

            return _warp_ld_a_fp64(A_local_buf, A_region, ki, thread_binding, rk)

        warp_row_tiles = self.warp_row_tiles
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        a_transposed = self.a_transposed
        # ldmatrix cannot be used for int8 + trans case.
        ldmatrix_available = not (DataType(a_dtype).bits != 16 and a_transposed)

        def mma_load_layout(i, j):
            return i, j

        if not ldmatrix_available:
            if DataType(a_dtype).bits == 4:
                mma_load_layout = mma_load_a_32x32_to_shared_16x64_layout
            elif DataType(a_dtype).bits == 8:
                mma_load_layout = mma_load_a_32x16_to_shared_16x32_layout
            elif DataType(a_dtype).bits == 16:
                mma_load_layout = mma_load_a_32x8_to_shared_16x16_layout
            elif DataType(a_dtype).bits == 32:
                mma_load_layout = mma_load_a_32x4_to_shared_16x8_layout
            else:
                raise ValueError(f"Unsupported dtype: {a_dtype}")

        thread_binding = self.get_thread_binding()

        # legalize shared buffer to region
        A_region = self._legalize_to_buffer_region(A_shared_buf)
        A_buf = A_region.buffer
        A_base0 = A_region.region[-2].min
        A_base1 = A_region.region[-1].min
        A_other = [r.min for r in A_region.region[:-2]]
        A_stride_last = A_buf.shape[-1]

        @T.macro
        def _warp_ldmatrix_a(
            A_local_buf,
            A_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            stride = A_stride_last
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            trans = self.a_transposed

            for i in T.serial(warp_rows):
                wi, wk = warp_m * warp_row_tiles + i * micro_size_x, rk * chunk + ki * micro_size_k

                if ldmatrix_available:
                    row_off, col_off = get_ldmatrix_offset("A", tx, 0, stride, a_dtype, a_transposed)
                    src_indices = (
                        tuple(A_other) + (A_base0 + wk + row_off, A_base1 + wi + col_off)
                        if a_transposed
                        else tuple(A_other) + (A_base0 + wi + row_off, A_base1 + wk + col_off)
                    )
                    T.ptx_ldmatrix(
                        T.bool(trans),
                        4,
                        T.access_ptr(A_buf[src_indices], "r", extent=8),
                        T.access_ptr(A_local_buf[i * local_size_a], "w", extent=8),
                    )
                else:
                    for j in T.serial(local_size_a):
                        mi, mk = mma_load_layout(tx, j)
                        if a_transposed:
                            A_local_buf[i * local_size_a + j] = A_buf[tuple(A_other) + (A_base0 + wk + mk, A_base1 + wi + mi)]
                        else:
                            A_local_buf[i * local_size_a + j] = A_buf[tuple(A_other) + (A_base0 + wi + mi, A_base1 + wk + mk)]

        return _warp_ldmatrix_a(A_local_buf, A_region, ki, thread_binding, rk)

    def ldmatrix_b(self, B_local_buf: Buffer, B_shared_buf: Buffer | BufferRegion, ki: PrimExpr, rk: PrimExpr | None = 0):
        # Fast path for fp64: no ldmatrix support, do direct per-lane loads
        b_dtype = self.b_dtype
        if DataType(b_dtype).bits == 64:
            warp_col_tiles = self.warp_col_tiles
            warp_cols = self.warp_cols
            chunk = self.chunk
            micro_size_y = self.micro_size_y  # 8
            micro_size_k = self.micro_size_k  # 4
            local_size_b = self.local_size_b  # 1
            b_transposed = self.b_transposed
            thread_binding = self.get_thread_binding()

            # legalize shared buffer to region
            B_region = self._legalize_to_buffer_region(B_shared_buf)
            B_buf = B_region.buffer
            B_base0 = B_region.region[-2].min
            B_base1 = B_region.region[-1].min
            B_other = [r.min for r in B_region.region[:-2]]

            @T.macro
            def _warp_ld_b_fp64(
                B_local_buf,
                B_shared_buf,
                ki,
                thread_binding,
                rk=0,
            ):
                tx, warp_n, _ = self.extract_thread_binding(thread_binding)
                for j in T.serial(warp_cols):
                    wi = warp_n * warp_col_tiles + j * micro_size_y
                    wk = rk * chunk + ki * micro_size_k
                    mi = tx // micro_size_k
                    mk = tx % micro_size_k
                    if b_transposed:
                        B_local_buf[j * local_size_b] = B_buf[tuple(B_other) + (B_base0 + wi + mi, B_base1 + wk + mk)]
                    else:
                        B_local_buf[j * local_size_b] = B_buf[tuple(B_other) + (B_base0 + wk + mk, B_base1 + wi + mi)]

            return _warp_ld_b_fp64(B_local_buf, B_region, ki, thread_binding, rk)

        warp_col_tiles = self.warp_col_tiles
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        b_transposed = self.b_transposed
        thread_binding = self.get_thread_binding()

        # legalize shared buffer to region
        B_region = self._legalize_to_buffer_region(B_shared_buf)
        B_buf = B_region.buffer
        B_base0 = B_region.region[-2].min
        B_base1 = B_region.region[-1].min
        B_other = [r.min for r in B_region.region[:-2]]
        B_stride_last = B_buf.shape[-1]
        replicate_b = self.n_dim == 16
        # ldmatrix cannot be used for int8 + trans case.
        ldmatrix_available = not (DataType(b_dtype).bits != 16 and not b_transposed)

        def mma_load_layout(i, j):
            return i, j

        if not ldmatrix_available:
            if DataType(b_dtype).bits == 4:
                mma_load_layout = mma_load_b_32x32_to_shared_16x64_layout
            elif DataType(b_dtype).bits == 8:
                mma_load_layout = mma_load_b_32x16_to_shared_16x32_layout
            elif DataType(b_dtype).bits == 16:
                mma_load_layout = mma_load_b_32x8_to_shared_16x16_layout
            elif DataType(b_dtype).bits == 32:
                mma_load_layout = mma_load_b_32x4_to_shared_16x8_layout
            else:
                raise ValueError(f"Unsupported dtype: {b_dtype}")

        @T.macro
        def _warp_ldmatrix_b(
            B_local_buf,
            B_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            stride = B_stride_last
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)
            trans = not b_transposed

            for i in T.serial(warp_cols):
                # Assign B_shared_elem
                wi, wk = (
                    warp_n * warp_col_tiles + i * micro_size_y,
                    rk * chunk + ki * micro_size_k,
                )

                if ldmatrix_available:
                    num = 4 if replicate_b else 2
                    row_off, col_off = get_ldmatrix_offset("B", tx, 0, stride, b_dtype, b_transposed)
                    src_indices = (
                        tuple(B_other) + (B_base0 + wi + row_off, B_base1 + wk + col_off)
                        if b_transposed
                        else tuple(B_other) + (B_base0 + wk + row_off, B_base1 + wi + col_off)
                    )
                    T.ptx_ldmatrix(
                        T.bool(trans),
                        num,
                        T.access_ptr(B_buf[src_indices], "r", extent=2 * num),
                        T.access_ptr(B_local_buf[i * local_size_b], "w", extent=2 * num),
                    )

                else:
                    # load 16x32 data from shared buffer to local buffer
                    # must be transposed.
                    for j in T.serial(local_size_b):
                        mi, mk = mma_load_layout(tx, j)
                        if b_transposed:
                            B_local_buf[i * local_size_b + j] = B_buf[tuple(B_other) + (B_base0 + wi + mi, B_base1 + wk + mk)]
                        else:
                            B_local_buf[i * local_size_b + j] = B_buf[tuple(B_other) + (B_base0 + wk + mk, B_base1 + wi + mi)]

        return _warp_ldmatrix_b(B_local_buf, B_shared_buf, ki, thread_binding, rk)

    def mma(self, A_local_buf: Buffer, B_local_buf: Buffer, C_local_buf: Buffer, k_inner: PrimExpr | None = 0):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols

        @T.macro
        def _warp_mma(A_local_buf, B_local_buf, C_local_buf):
            for i, j in T.grid(warp_rows, warp_cols):
                self.mma_atom(A_local_buf, B_local_buf, C_local_buf, i, j, k_inner)

        return _warp_mma(A_local_buf, B_local_buf, C_local_buf)

    # ---- Atom-level interface ----

    @property
    def mma_num_inst_m(self) -> int:
        """Number of MMA instruction atoms along the M dimension."""
        return self.warp_rows

    @property
    def mma_num_inst_n(self) -> int:
        """Number of MMA instruction atoms along the N dimension."""
        return self.warp_cols

    def mma_atom(
        self,
        A_local_buf: Buffer,
        B_local_buf: Buffer,
        C_local_buf: Buffer,
        inst_m_idx: PrimExpr | int,
        inst_n_idx: PrimExpr | int,
        k_inner: PrimExpr | int = 0,
    ):
        """Emit a single MMA atom for tile (inst_m_idx, inst_n_idx).

        This is the atomic building block of ``mma()``.  Calling this method
        for every ``(i, j)`` in ``T.grid(mma_num_inst_m, mma_num_inst_n)``
        produces identical TIR to a single ``mma()`` call.

        Parameters
        ----------
        A_local_buf : Buffer
            Fragment buffer for operand A.
        B_local_buf : Buffer
            Fragment buffer for operand B.
        C_local_buf : Buffer
            Accumulator fragment buffer.
        inst_m_idx : int or PrimExpr
            M-dimension atom index (0 .. mma_num_inst_m - 1).
        inst_n_idx : int or PrimExpr
            N-dimension atom index (0 .. mma_num_inst_n - 1).
        k_inner : int or PrimExpr
            K-inner step index used to offset A/B fragments.
        """
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        accum_dtype = self.accum_dtype
        accum_dtype_abbrv = self.accum_dtype_abbrv
        mma_prefix = self.mma_prefix
        replicate_b = self.n_dim == 16

        a_is_fragment = is_fragment(A_local_buf)
        b_is_fragment = is_fragment(B_local_buf)
        a_local_stride: PrimExpr = k_inner * warp_rows * local_size_a if a_is_fragment else 0
        b_local_stride: PrimExpr = k_inner * warp_cols * local_size_b if b_is_fragment else 0

        A_offset = a_local_stride + inst_m_idx * local_size_a
        B_offset = b_local_stride + inst_n_idx * local_size_b
        C_offset = inst_m_idx * warp_cols * local_size_out + inst_n_idx * local_size_out

        @T.macro
        def _atom_mma(A_local_buf, B_local_buf, C_local_buf):
            T.ptx_mma(
                accum_dtype,
                mma_prefix,
                "row",
                "col",
                a_dtype_abbrv,
                b_dtype_abbrv,
                accum_dtype_abbrv,
                A_local_buf.data,
                A_offset,
                B_local_buf.data,
                B_offset,
                C_local_buf.data,
                C_offset,
                T.bool(False),
            )
            if replicate_b:
                T.ptx_mma(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    accum_dtype_abbrv,
                    A_local_buf.data,
                    A_offset,
                    B_local_buf.data,
                    B_offset + lift(local_size_b) // 2,
                    C_local_buf.data,
                    C_offset + lift(local_size_out) // 2,
                    T.bool(False),
                )

        return _atom_mma(A_local_buf, B_local_buf, C_local_buf)

    def stmatrix(self, C_local_buf, C_buf, pid_m=None, pid_n=None):
        block_row_warps = self.block_row_warps
        block_col_warps = self.block_col_warps
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_out = self.local_size_out

        is_global = pid_m is not None and pid_n is not None
        BLOCK_M = block_row_warps * warp_rows
        BLOCK_N = block_col_warps * warp_cols
        M_DIM, n_dim = self.M_DIM, self.n_dim
        C_buf_dims = len(C_buf.shape)
        assert C_buf_dims in {2, 4}, "C_buf should be 2D or 4D"

        thread_binding = self.get_thread_binding()
        store_index_map = mma_store_index_map_fp64 if self._use_fp64_store_index_map() else mma_store_index_map

        # STS
        # MMA Store must be in simulated instead of TVM Intrins
        # As TVM Intrins is like a hack that the threadIdx.x should be always
        # equal to the warp_size
        @T.macro
        def _warp_stmatrix_shared(C_local_buf, C_buf, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            for i, j in T.grid(warp_rows, warp_cols):
                for local_id_o in T.serial(local_size_out // 2):
                    for local_id_i in T.vectorized(2):
                        local_id = local_id_o * 2 + local_id_i
                        row, col = T.meta_var(store_index_map(tx, local_id))
                        if C_buf_dims == 2:
                            C_buf[(warp_m * warp_rows + i) * M_DIM + row, (warp_n * warp_cols + j) * n_dim + col] = C_local_buf[
                                i * (warp_cols * local_size_out) + j * local_size_out + local_id
                            ]
                        else:
                            C_buf[warp_m * warp_rows + i, warp_n * warp_cols + j, row, col] = C_local_buf[
                                i * (warp_cols * local_size_out) + j * local_size_out + local_id
                            ]

        @T.macro
        def _warp_stmatrix_global(C_local_buf, C_buf, thread_binding):
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)
            for i, j in T.grid(warp_rows, warp_cols):
                for local_id_o in T.serial(local_size_out // 2):
                    for local_id_i in T.vectorized(2):
                        local_id = local_id_o * 2 + local_id_i
                        row, col = T.meta_var(store_index_map(tx, local_id))
                        C_buf[
                            (pid_m * BLOCK_M + warp_m * warp_rows + i) * M_DIM + row,
                            (pid_n * BLOCK_N + warp_n * warp_cols + j) * n_dim + col,
                        ] = C_local_buf[i * warp_cols * local_size_out + j * local_size_out + local_id]

        return (
            _warp_stmatrix_global(C_local_buf, C_buf, thread_binding)
            if is_global
            else _warp_stmatrix_shared(C_local_buf, C_buf, thread_binding)
        )

    def make_mma_load_layout(self, local_buf: Buffer, matrix: Literal["A", "B"] = "A") -> T.Fragment:
        """
        Create a layout function for storing MMA results into a fragment buffer.
        This layout is used in conjunction with `inverse_mma_store_layout` to
        map fragment indices to threads and local indices.

        Parameters
        ----------
        local_buf : tirx.Buffer
            The local buffer representing a fragment of a matrix.

        Returns
        -------
        T.Fragment
            A fragment object that describes how threads and indices
            in `local_buf` are laid out.

        Raises
        ------
        AssertionError
            If `local_buf` is not detected to be a fragment buffer.
        """
        from tilelang.utils import is_fragment

        assert matrix in ["A", "B"], "matrix should be either A or B"
        matrix_is_a: bool = matrix == "A"
        matrix_is_b: bool = matrix == "B"
        dtype = self.a_dtype if matrix_is_a else self.b_dtype
        dtype_bits = DataType(dtype).bits
        transposed = self.a_transposed if matrix_is_a else self.b_transposed

        # s represents spatial axis
        # r represents reduction axis
        # sr represents the two dims are spatial + reduction
        # rs represents the two dims are reduction + spatial
        # sr also can represent a non-transposed basic layout
        # then rs also can represent a transposed basic layout
        transform_func_sr_a: Callable = None
        transform_func_sr_b: Callable = None
        if dtype_bits == 32:
            transform_func_sr_a = shared_16x8_to_mma_32x4_layout_sr_a
            transform_func_sr_b = shared_16x8_to_mma_32x4_layout_sr_b
        elif dtype_bits == 16:
            transform_func_sr_a = shared_16x16_to_mma_32x8_layout_sr_a
            transform_func_sr_b = shared_16x16_to_mma_32x8_layout_sr_b
        elif dtype_bits == 8:
            transform_func_sr_a = shared_16x32_to_mma_32x16_layout_sr_a
            transform_func_sr_b = shared_16x32_to_mma_32x16_layout_sr_b
        else:
            raise ValueError(f"Unsupported dtype {dtype}")

        is_sr_conditions = [False]
        is_sr_conditions.append(matrix_is_a and not transposed)
        is_sr_conditions.append(matrix_is_b and transposed)
        is_sr_axis_order = any(is_sr_conditions)

        # the layout of mma.sync is row.col.
        # so the b matrix expected a transposed basic layout
        transform_func: Callable = None
        if matrix_is_a:
            transform_func = transform_func_sr_a if is_sr_axis_order else lambda i, j: transform_func_sr_a(j, i)
        elif matrix_is_b:
            transform_func = transform_func_sr_b if is_sr_axis_order else lambda i, j: transform_func_sr_b(j, i)
        else:
            raise ValueError(f"Unsupported matrix {matrix}")

        assert is_fragment(local_buf), f"local_buf must be a fragment, but got {local_buf.scope()}"

        if matrix_is_a:
            micro_size_s, micro_size_r = self.micro_size_x, self.micro_size_k
        else:
            micro_size_r, micro_size_s = self.micro_size_k, self.micro_size_y

        block_row_warps, block_col_warps = (
            self.block_row_warps,
            self.block_col_warps,
        )

        inverse_mma_load_layout = IndexMap.from_func(transform_func, index_dtype=T.int32)

        def forward_thread(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            """
            lane_id, _ = inverse_mma_load_layout.map_indices([i, j])
            return lane_id

        def forward_index(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            """
            _, local_id = inverse_mma_load_layout.map_indices([i, j])
            return local_id

        base_fragment = T.Fragment(
            [micro_size_s, micro_size_r] if is_sr_axis_order else [micro_size_r, micro_size_s],
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )

        warp_rows, warp_cols = self.warp_rows, self.warp_cols
        chunk = self.chunk

        warp_s = warp_rows if matrix_is_a else warp_cols
        warp_r = chunk // micro_size_r
        block_s = block_row_warps if matrix_is_a else block_col_warps
        replicate = block_col_warps if matrix_is_a else block_row_warps

        if is_sr_axis_order:
            warp_fragment = base_fragment.repeat([warp_s, warp_r], repeat_on_thread=False, lower_dim_first=False)
            if matrix_is_a:
                block_fragment = warp_fragment.repeat([block_s, 1], repeat_on_thread=True, lower_dim_first=True).replicate(replicate)
            elif matrix_is_b:
                block_fragment = warp_fragment.replicate(replicate).repeat([block_s, 1], repeat_on_thread=True, lower_dim_first=True)
            else:
                raise ValueError(f"Unsupported matrix type {matrix}")
        else:
            warp_fragment = base_fragment.repeat([warp_r, warp_s], repeat_on_thread=False, lower_dim_first=True)
            if matrix_is_a:
                block_fragment = warp_fragment.repeat([1, block_s], repeat_on_thread=True, lower_dim_first=True).replicate(replicate)
            elif matrix_is_b:
                block_fragment = warp_fragment.replicate(replicate).repeat([1, block_s], repeat_on_thread=True, lower_dim_first=True)
            else:
                raise ValueError(f"Unsupported matrix type {matrix}")

        return block_fragment

    def make_mma_store_layout(self, local_buf: Buffer) -> T.Fragment:
        """
        Create a layout function for storing MMA results into a fragment buffer.
        This layout is used in conjunction with `inverse_mma_store_layout` to
        map fragment indices to threads and local indices.

        Parameters
        ----------
        local_buf : tirx.Buffer
            The local buffer representing a fragment of a matrix.

        Returns
        -------
        T.Fragment
            A fragment object that describes how threads and indices
            in `local_buf` are laid out.

        Raises
        ------
        AssertionError
            If `local_buf` is not detected to be a fragment buffer.
        """
        from tilelang.utils import is_fragment

        shape = local_buf.shape
        assert is_fragment(local_buf), f"local_buf {local_buf} must be a fragment, but got {local_buf.scope()}"
        inverse_mma_store_layout = self.get_store_index_map(inverse=True)

        micro_size_x, micro_size_y = self.micro_size_x, self.micro_size_y
        local_size_out = self.local_size_out
        block_row_warps, block_col_warps = self.block_row_warps, self.block_col_warps
        warp_rows, warp_cols = self.warp_rows, self.warp_cols
        warp_size = self.WARP_SIZE
        is_m_first = self.is_m_first

        def forward_thread(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            map them to a thread index according to `inverse_mma_store_layout`.
            """
            # the upper bounds of i and j are block_row_warps * warp_rows * micro_size_x and block_col_warps * warp_cols * micro_size_y
            # the upper bounds of block_row_warps and block_col_warps are warp_rows and warp_cols
            block_i, block_j = (i // micro_size_x) // warp_rows, (j // micro_size_y) // warp_cols
            # upper bounds of mma_i and mma_j are micro_size_x and micro_size_y
            mma_i, mma_j = i % micro_size_x, j % micro_size_y
            lane_id, _ = inverse_mma_store_layout.map_indices([mma_i, mma_j])
            if is_m_first:
                thread_id = block_i * (block_col_warps * warp_cols) + block_j * warp_size + lane_id
            else:
                thread_id = block_j * (block_row_warps * warp_size) + block_i * warp_size + lane_id
            return thread_id

        def forward_index(i: int, j: int) -> int:
            """
            Given the row index `i` and column index `j` in the fragment,
            map them to a local index in a single thread according
            to `inverse_mma_store_layout`.
            """
            # the upper bounds of i and j are block_row_warps * warp_rows * micro_size_x and block_col_warps * warp_cols * micro_size_y
            # the upper bounds of warp_i and warp_j are warp_rows and warp_cols
            warp_i, warp_j = (i // micro_size_x) % warp_rows, (j // micro_size_y) % warp_cols
            # upper bounds of mma_i and mma_j are micro_size_x and micro_size_y
            mma_i, mma_j = i % micro_size_x, j % micro_size_y
            _, local_id = inverse_mma_store_layout.map_indices([mma_i, mma_j])
            return warp_i * (warp_cols * local_size_out) + warp_j * local_size_out + local_id

        return T.Fragment(
            shape,
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )

    @staticmethod
    def _legalize_to_buffer_region(obj: Buffer | BufferLoad | BufferRegion) -> BufferRegion:
        """
        Convert Buffer/BufferRegion/BufferLoad to a BufferRegion.

        - Buffer -> full-region BufferRegion covering entire shape
        - BufferRegion -> returned as-is
        - BufferLoad -> best-effort convert via get_buffer_region_from_load;
        if scalar, fall back to 1-sized ranges at given indices
        """
        if isinstance(obj, BufferRegion):
            return obj
        if isinstance(obj, Buffer):
            mins = [tirx.IntImm("int32", 0) for _ in obj.shape]
            ranges = [Range.from_min_extent(m, e) for m, e in zip(mins, obj.shape)]
            return BufferRegion(obj, ranges)
        if isinstance(obj, BufferLoad):
            region = get_buffer_region_from_load(obj)
            if region is not None:
                return region
            # Fallback: scalar load -> 1-sized ranges at indices
            mins = [idx for idx in obj.indices]
            ones = [tirx.IntImm("int32", 1) for _ in obj.indices]
            ranges = [Range.from_min_extent(m, e) for m, e in zip(mins, ones)]
            return BufferRegion(obj.buffer, ranges)
        raise ValueError(f"Unsupported argument type for BufferRegion: {type(obj)}")


# ================================================================
# PPU Subclass — PPUTensorCoreIntrinEmitter
# ================================================================


class PPUTensorCoreIntrinEmitter(TensorCoreIntrinEmitter):
    """
    PPU-specific overrides of TensorCoreIntrinEmitter.

    Only methods that differ from the base class (TensorCoreIntrinEmitter) are defined here.
    Inherits all other behavior (stmatrix, mma, make_mma_store_layout, etc.)
    from the base class unchanged.
    """

    def __init__(self, *args, ppu_arch=None, **kwargs):
        if ppu_arch is None:
            from tilelang.contrib import hgcc

            compute_version = hgcc.get_target_compute_version()
            arch_str = hgcc.get_target_arch(compute_version)
            ppu_arch = int(arch_str.split("_")[-1].rstrip("af"))
        self.ppu_arch = ppu_arch
        super().__init__(*args, **kwargs)

    def _initialize_mma_prefix(self, k_dim: int = 16):
        a_dtype = DataType(self.a_dtype)
        b_dtype = DataType(self.b_dtype)
        accum_dtype = DataType(self.accum_dtype)
        is_ppu0010_int8 = (
            self.ppu_arch == 10
            and a_dtype == DataType("int8")
            and b_dtype == DataType("int8")
            and accum_dtype == DataType("int32")
            and not self.a_transposed
        )
        if k_dim == 8:
            self.mma_prefix = "m16n16k8"
        elif k_dim == 16:
            self.mma_prefix = "m16n16k16"
        elif k_dim == 32 and (is_ppu0010_int8 or (self.ppu_arch >= 15 and not DataType(self.a_dtype).is_float4_e2m1fn())):
            self.mma_prefix = "m16n16k32"
        elif (
            k_dim == 64 and self.ppu_arch >= 15 and DataType(self.a_dtype).is_float4_e2m1fn() and DataType(self.b_dtype).is_float4_e2m1fn()
        ):
            # FP4 e2m1: PPU0015_16x16x64_F32F4F4F32_TN
            self.mma_prefix = "m16n16k64"
        else:
            raise ValueError(f"Unsupported k_dim {k_dim}")

    def _initialize_micro_size(self, m_dim: int = 16, k_dim: int = 16):
        warp_row_tiles = self.warp_row_tiles
        warp_col_tiles = self.warp_col_tiles
        assert warp_row_tiles >= 16, f"warp_row_tiles must be >= 16, got {warp_row_tiles}"
        assert warp_row_tiles % 16 == 0, f"warp_row_tiles must be divisible by 16, got {warp_row_tiles}"
        assert warp_col_tiles >= 16, f"warp_col_tiles must be >= 16, got {warp_col_tiles}"
        assert warp_col_tiles % 16 == 0, f"warp_col_tiles must be divisible by 16, got {warp_col_tiles}"
        self.warp_rows = warp_row_tiles // m_dim
        self.n_dim = 16
        self.micro_size_y = 16
        self.warp_cols = warp_col_tiles // 16
        self.micro_size_x = m_dim
        self.micro_size_k = k_dim

    def get_store_index_map(self, inverse: bool = False) -> IndexMap:
        warp_size, local_size_c = self.WARP_SIZE, self.local_size_out
        store_layout = ppu_mma_store_32x8_to_shared_16x16_layout
        if self.ppu_arch >= 15:
            store_layout = mma_store_32x8_to_shared_16x16_layout
        index_map = IndexMap.from_func(store_layout, index_dtype=T.int32)
        if not inverse:
            return index_map
        inverse_index_map = index_map.inverse([warp_size, local_size_c])
        return inverse_index_map

    def _disable_ldmat_swzl(self) -> bool:
        """Whether the swizzled ldmatrix variant should be disabled.

        Disabled when the pass-context flag disables it or the PPU arch
        is not 89 (only arch 89 supports the swizzle variant).
        """
        pass_ctx = get_pass_context()
        config_disabled = pass_ctx.config.get(PassConfigKey.TL_DISABLE_LDMAT_SWZL.value, False)
        return config_disabled or self.ppu_arch != 15

    def _ldmat_swzl_mode(self, stride, dtype: str) -> tuple[int, bool]:
        """Determine the swizzle mode and whether swizzle is supported for ``stride``.

        The mode is decided by the shared-memory row width in bytes:
        64 bytes -> mode 1 (half-bank), multiples of 128 bytes -> mode 0 (full-bank).
        """
        value = stride if isinstance(stride, int) else getattr(stride, "value", None)
        if value is None:
            return 0, False
        bytes_per_row = value * DataType(dtype).bits // 8
        if bytes_per_row == 64:
            return 1, True
        if bytes_per_row % 128 == 0:
            return 0, True
        return 0, False

    def ldmatrix_a(self, A_local_buf: Buffer, A_shared_buf: Buffer | BufferRegion, ki: PrimExpr, rk: PrimExpr | None = 0):
        a_dtype = DataType(self.a_dtype)
        a_bits = a_dtype.bits
        is_ppu0010_int8 = (
            self.ppu_arch == 10
            and a_dtype == DataType("int8")
            and DataType(self.b_dtype) == DataType("int8")
            and DataType(self.accum_dtype) == DataType("int32")
        )
        if a_bits not in (4, 8, 16):
            return self._ldmatrix_a_default(A_local_buf, A_shared_buf, ki, rk)
        if a_bits in (4, 8) and self.a_transposed:
            return self._ldmatrix_a_default(A_local_buf, A_shared_buf, ki, rk)

        warp_row_tiles = self.warp_row_tiles
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        a_transposed = self.a_transposed

        thread_binding = self.get_thread_binding()

        # legalize shared buffer to region
        A_region = self._legalize_to_buffer_region(A_shared_buf)
        A_buf = A_region.buffer
        A_base0 = A_region.region[-2].min
        A_base1 = A_region.region[-1].min
        A_other = [r.min for r in A_region.region[:-2]]

        swzl_mode, swzl_supported = self._ldmat_swzl_mode(A_buf.shape[-1], self.a_dtype)
        disable_ldmat_swzl = self._disable_ldmat_swzl() or not swzl_supported
        if a_bits == 4:
            access_extent = 32
        elif a_bits == 8:
            access_extent = 16
        else:
            access_extent = 8

        @T.macro
        def _warp_ldmatrix_a(
            A_local_buf,
            A_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            trans = self.a_transposed

            for i in T.serial(warp_rows):
                wi, wk = warp_m * warp_row_tiles + i * micro_size_x, rk * chunk + ki * micro_size_k
                if a_transposed:
                    row_off, col_off = ppu_ldmatrix_trans_32x8_to_shared_16x16_layout(tx, 0)
                    src_indices = tuple(A_other) + (A_base0 + wk + row_off, A_base1 + wi + col_off)
                else:
                    if a_bits == 4:
                        row_off, col_off = ldmatrix_32x16_to_shared_16x64_layout_a(tx, 0)
                    elif is_ppu0010_int8:
                        row_off, col_off = ppu_ldmatrix_32x16_to_shared_16x32_layout_s8_a(tx, 0)
                    elif a_bits == 8:
                        row_off, col_off = ldmatrix_32x16_to_shared_16x32_layout_a(tx, 0)
                    elif self.ppu_arch >= 15:
                        row_off, col_off = ldmatrix_32x8_to_shared_16x16_layout(tx, 0)
                    else:
                        row_off, col_off = ppu_ldmatrix_32x8_to_shared_16x16_layout(tx, 0)
                    src_indices = tuple(A_other) + (A_base0 + wi + row_off, A_base1 + wk + col_off)
                if disable_ldmat_swzl:
                    T.ptx_ldmatrix(
                        T.bool(trans),
                        4,
                        T.access_ptr(A_buf[src_indices], "r", extent=access_extent),
                        T.access_ptr(A_local_buf[i * local_size_a], "w", extent=access_extent),
                    )
                else:
                    T.tix_ldmatrix_swzl(
                        T.bool(trans),
                        4,
                        T.access_ptr(A_buf[src_indices], "r", extent=access_extent),
                        T.access_ptr(A_local_buf[i * local_size_a], "w", extent=access_extent),
                        swzl_mode,
                        a_transposed,
                    )

        return _warp_ldmatrix_a(A_local_buf, A_region, ki, thread_binding, rk)

    def _ldmatrix_a_default(self, A_local_buf: Buffer, A_shared_buf: Buffer | BufferRegion, ki: PrimExpr, rk: PrimExpr | None = 0):
        """Non-16-bit A loading with PPU arch-specific 32-bit layout."""
        a_dtype = self.a_dtype

        warp_row_tiles = self.warp_row_tiles
        warp_rows = self.warp_rows
        chunk = self.chunk
        micro_size_x = self.micro_size_x
        micro_size_k = self.micro_size_k
        local_size_a = self.local_size_a
        a_transposed = self.a_transposed
        # ldmatrix cannot be used for int8 + trans case.
        ldmatrix_available = not (DataType(a_dtype).bits != 16 and a_transposed)

        def mma_load_layout(i, j):
            return i, j

        if not ldmatrix_available:
            if DataType(a_dtype).bits == 4:
                mma_load_layout = mma_load_a_32x32_to_shared_16x64_layout
            elif DataType(a_dtype).bits == 8:
                mma_load_layout = mma_load_a_32x16_to_shared_16x32_layout
            elif DataType(a_dtype).bits == 16:
                mma_load_layout = mma_load_a_32x8_to_shared_16x16_layout
            elif DataType(a_dtype).bits == 32:
                if self.ppu_arch == 10:
                    mma_load_layout = ppu_mma_load_a_32x4_to_shared_16x8_layout
                else:
                    mma_load_layout = mma_load_a_32x4_to_shared_16x8_layout
            else:
                raise ValueError(f"Unsupported dtype: {a_dtype}")

        thread_binding = self.get_thread_binding()

        # legalize shared buffer to region
        A_region = self._legalize_to_buffer_region(A_shared_buf)
        A_buf = A_region.buffer
        A_base0 = A_region.region[-2].min
        A_base1 = A_region.region[-1].min
        A_other = [r.min for r in A_region.region[:-2]]
        A_stride_last = A_buf.shape[-1]

        @T.macro
        def _warp_ldmatrix_a(
            A_local_buf,
            A_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            stride = A_stride_last
            tx, _, warp_m = self.extract_thread_binding(thread_binding)
            trans = self.a_transposed

            for i in T.serial(warp_rows):
                wi, wk = warp_m * warp_row_tiles + i * micro_size_x, rk * chunk + ki * micro_size_k

                if ldmatrix_available:
                    if DataType(a_dtype).bits == 32 and self.ppu_arch == 10:
                        row_off, col_off = ppu_ldmatrix_32x4_to_shared_16x8_layout_a(tx, 0)
                    else:
                        row_off, col_off = get_ldmatrix_offset("A", tx, 0, stride, a_dtype, a_transposed)
                    src_indices = (
                        tuple(A_other) + (A_base0 + wk + row_off, A_base1 + wi + col_off)
                        if a_transposed
                        else tuple(A_other) + (A_base0 + wi + row_off, A_base1 + wk + col_off)
                    )
                    T.ptx_ldmatrix(
                        T.bool(trans),
                        4,
                        T.access_ptr(A_buf[src_indices], "r", extent=8),
                        T.access_ptr(A_local_buf[i * local_size_a], "w", extent=8),
                    )
                else:
                    for j in T.serial(local_size_a):
                        mi, mk = mma_load_layout(tx, j)
                        if a_transposed:
                            A_local_buf[i * local_size_a + j] = A_buf[tuple(A_other) + (A_base0 + wk + mk, A_base1 + wi + mi)]
                        else:
                            A_local_buf[i * local_size_a + j] = A_buf[tuple(A_other) + (A_base0 + wi + mi, A_base1 + wk + mk)]

        return _warp_ldmatrix_a(A_local_buf, A_region, ki, thread_binding, rk)

    def ldmatrix_b(
        self, B_local_buf: Buffer, B_shared_buf: Buffer | BufferRegion, ki: PrimExpr, rk: PrimExpr | None = 0, a_from_gemm_c: bool = False
    ):
        b_bits = DataType(self.b_dtype).bits
        if b_bits not in (4, 8, 16):
            return self._ldmatrix_b_default(B_local_buf, B_shared_buf, ki, rk)
        if b_bits in (4, 8) and not self.b_transposed:
            return self._ldmatrix_b_default(B_local_buf, B_shared_buf, ki, rk)

        warp_col_tiles = self.warp_col_tiles
        warp_cols = self.warp_cols
        chunk = self.chunk
        micro_size_y = self.micro_size_y
        micro_size_k = self.micro_size_k
        local_size_b = self.local_size_b
        b_transposed = self.b_transposed
        thread_binding = self.get_thread_binding()

        # legalize shared buffer to region
        B_region = self._legalize_to_buffer_region(B_shared_buf)
        B_buf = B_region.buffer
        B_base0 = B_region.region[-2].min
        B_base1 = B_region.region[-1].min
        B_other = [r.min for r in B_region.region[:-2]]
        replicate_b = self.n_dim == 16

        swzl_mode, swzl_supported = self._ldmat_swzl_mode(B_buf.shape[-1], self.b_dtype)
        disable_ldmat_swzl = self._disable_ldmat_swzl() or not swzl_supported

        @T.macro
        def _warp_ldmatrix_b(
            B_local_buf,
            B_shared_buf,
            ki,
            thread_binding,
            rk=0,
        ):
            tx, warp_n, _ = self.extract_thread_binding(thread_binding)
            trans = not b_transposed

            for i in T.serial(warp_cols):
                # Assign B_shared_elem
                wi, wk = (
                    warp_n * warp_col_tiles + i * micro_size_y,
                    rk * chunk + ki * micro_size_k,
                )

                num = 4 if replicate_b else 2
                access_extent = (2 * num) * (16 // b_bits)
                if b_bits == 4:
                    row_off, col_off = ldmatrix_32x16_to_shared_16x64_layout_b(tx, 0)
                elif b_bits == 8:
                    row_off, col_off = ldmatrix_32x16_to_shared_16x32_layout_b(tx, 0)
                elif self.ppu_arch >= 15 and not b_transposed:
                    row_off, col_off = ldmatrix_32x8_to_shared_16x16_layout(tx, 0)
                else:
                    row_off, col_off = ppu_ldmatrix_32x8_to_shared_16x16_layout(tx, 0)
                if a_from_gemm_c and not b_transposed and self.ppu_arch < 15:
                    src_row_off = (row_off // 8) * 8 + (row_off % 2) * 4 + ((row_off % 8) // 2)
                else:
                    src_row_off = row_off
                src_indices = (
                    tuple(B_other) + (B_base0 + wi + src_row_off, B_base1 + wk + col_off)
                    if b_transposed
                    else tuple(B_other) + (B_base0 + wk + src_row_off, B_base1 + wi + col_off)
                )
                if disable_ldmat_swzl or num != 4:
                    T.ptx_ldmatrix(
                        T.bool(trans),
                        num,
                        T.access_ptr(B_buf[src_indices], "r", extent=access_extent),
                        T.access_ptr(B_local_buf[i * local_size_b], "w", extent=access_extent),
                    )
                else:
                    T.tix_ldmatrix_swzl(
                        T.bool(trans),
                        num,
                        T.access_ptr(B_buf[src_indices], "r", extent=access_extent),
                        T.access_ptr(B_local_buf[i * local_size_b], "w", extent=access_extent),
                        swzl_mode,
                        b_transposed,
                    )

        return _warp_ldmatrix_b(B_local_buf, B_shared_buf, ki, thread_binding, rk)

    def _ldmatrix_b_default(self, B_local_buf: Buffer, B_shared_buf: Buffer | BufferRegion, ki: PrimExpr, rk: PrimExpr | None = 0):
        """Non-16-bit B loading — delegates to base class (TensorCoreIntrinEmitter) ldmatrix_b."""
        return TensorCoreIntrinEmitter.ldmatrix_b(self, B_local_buf, B_shared_buf, ki, rk)

    def mma(
        self,
        A_local_buf: Buffer,
        B_local_buf: Buffer,
        C_local_buf: Buffer,
        k_inner: PrimExpr | None = 0,
        scale_A: Buffer | BufferRegion | None = None,
        scale_B: Buffer | BufferRegion | None = None,
    ):
        """Issue PPU MMA atoms, optionally with runtime MXFP4 E8M0 scales."""
        if (scale_A is None) != (scale_B is None):
            raise ValueError("scale_A and scale_B must be provided together")

        warp_rows = self.warp_rows
        warp_cols = self.warp_cols

        @T.macro
        def _warp_mma(A_local_buf, B_local_buf, C_local_buf):
            for i, j in T.grid(warp_rows, warp_cols):
                self.mma_atom(
                    A_local_buf,
                    B_local_buf,
                    C_local_buf,
                    i,
                    j,
                    k_inner,
                    scale_A,
                    scale_B,
                )

        return _warp_mma(A_local_buf, B_local_buf, C_local_buf)

    def mma_atom(
        self,
        A_local_buf: Buffer,
        B_local_buf: Buffer,
        C_local_buf: Buffer,
        inst_m_idx: PrimExpr | int,
        inst_n_idx: PrimExpr | int,
        k_inner: PrimExpr | int = 0,
        scale_A: Buffer | BufferRegion | None = None,
        scale_B: Buffer | BufferRegion | None = None,
    ):
        warp_rows = self.warp_rows
        warp_cols = self.warp_cols
        local_size_a = self.local_size_a
        local_size_b = self.local_size_b
        local_size_out = self.local_size_out
        a_dtype_abbrv = self.a_dtype_abbrv
        b_dtype_abbrv = self.b_dtype_abbrv
        accum_dtype = self.accum_dtype
        accum_dtype_abbrv = self.accum_dtype_abbrv
        mma_prefix = self.mma_prefix
        # PPU uses m16n16 instructions, so B does not need replication.

        a_is_fragment = is_fragment(A_local_buf)
        b_is_fragment = is_fragment(B_local_buf)
        a_local_stride: PrimExpr = k_inner * warp_rows * local_size_a if a_is_fragment else 0
        b_local_stride: PrimExpr = k_inner * warp_cols * local_size_b if b_is_fragment else 0

        A_offset = a_local_stride + inst_m_idx * local_size_a
        B_offset = b_local_stride + inst_n_idx * local_size_b
        C_offset = inst_m_idx * warp_cols * local_size_out + inst_n_idx * local_size_out

        scale_operands = None
        if scale_A is not None:
            if self.a_dtype_abbrv != "e2m1" or self.b_dtype_abbrv != "e2m1" or self.micro_size_k != 64:
                raise ValueError("runtime E8M0 scales are only supported for PPU FP4 m16n16k64 MMA")

            scale_A_region = self._legalize_to_buffer_region(scale_A)
            scale_B_region = self._legalize_to_buffer_region(scale_B)
            if not scale_A_region.region or not scale_B_region.region:
                raise ValueError("PPU MXFP4 scale regions must have a row or column dimension")

            scale_A_buf = scale_A_region.buffer
            scale_B_buf = scale_B_region.buffer
            scale_A_base = scale_A_region.region[-1].min
            scale_B_base = scale_B_region.region[-1].min
            scale_k_tiles = self.chunk // self.micro_size_k
            thread_binding = self.get_thread_binding()
            tx, warp_n, warp_m = self.extract_thread_binding(thread_binding)

            # The PPU0015 scale collective packs the two rows/columns owned by
            # a lane into the low/high uint16 halves of S0/S1.  Each uint16 is
            # itself low-byte-first for the two block-32 scales.  S2/S3 select
            # one of the four 16-row/column atom groups supplied by lane%4.
            lane_group = tx % 4
            lane_in_group = tx // 4
            # Warp tiles wider than four atoms along an operand dimension are
            # handled in chunks of four atom groups: each chunk re-arms the
            # scale registers with its own 64-row/column band and the atom
            # selects its part modulo four.  Within each chunk, all 32 lanes
            # participate in the scale collective even when the chunk covers
            # fewer than four atom groups; redirect unused lane groups to the
            # chunk's first group so they never read past a compact scale
            # region.  The redirected parts are never selected by the bounded
            # MMA atom loops.
            if warp_rows <= 4:
                scale_A_chunk = 0
                scale_A_selector = inst_m_idx
            else:
                scale_A_chunk = inst_m_idx // 4 * 4
                scale_A_selector = inst_m_idx % 4
            if warp_cols <= 4:
                scale_B_chunk = 0
                scale_B_selector = inst_n_idx
            else:
                scale_B_chunk = inst_n_idx // 4 * 4
                scale_B_selector = inst_n_idx % 4
            scale_A_group = T.if_then_else(lane_group < warp_rows - scale_A_chunk, lane_group, 0)
            scale_B_group = T.if_then_else(lane_group < warp_cols - scale_B_chunk, lane_group, 0)
            scale_A_index = scale_A_base + warp_m * self.warp_row_tiles + (scale_A_chunk + scale_A_group) * 16 + lane_in_group
            scale_B_index = scale_B_base + warp_n * self.warp_col_tiles + (scale_B_chunk + scale_B_group) * 16 + lane_in_group
            if scale_k_tiles == 1:
                scale_A_prefix = tuple(r.min for r in scale_A_region.region[:-1])
                scale_B_prefix = tuple(r.min for r in scale_B_region.region[:-1])
                scale_A_coords = scale_A_prefix + (scale_A_index,)
                scale_B_coords = scale_B_prefix + (scale_B_index,)
                scale_A_coords_hi = scale_A_prefix + (scale_A_index + 8,)
                scale_B_coords_hi = scale_B_prefix + (scale_B_index + 8,)
            else:
                scale_A_prefix = tuple(r.min for r in scale_A_region.region[:-2])
                scale_B_prefix = tuple(r.min for r in scale_B_region.region[:-2])
                scale_A_k = scale_A_region.region[-2].min + k_inner
                scale_B_k = scale_B_region.region[-2].min + k_inner
                scale_A_coords = scale_A_prefix + (scale_A_k, scale_A_index)
                scale_B_coords = scale_B_prefix + (scale_B_k, scale_B_index)
                scale_A_coords_hi = scale_A_prefix + (scale_A_k, scale_A_index + 8)
                scale_B_coords_hi = scale_B_prefix + (scale_B_k, scale_B_index + 8)
            scale_A_reg = T.bitwise_or(
                T.cast(scale_A_buf[scale_A_coords], T.uint32),
                T.shift_left(T.cast(scale_A_buf[scale_A_coords_hi], T.uint32), 16),
            )
            scale_B_reg = T.bitwise_or(
                T.cast(scale_B_buf[scale_B_coords], T.uint32),
                T.shift_left(T.cast(scale_B_buf[scale_B_coords_hi], T.uint32), 16),
            )
            scale_operands = (scale_A_reg, scale_B_reg, scale_A_selector, scale_B_selector)

        if scale_operands is None:

            @T.macro
            def _atom_mma(A_local_buf, B_local_buf, C_local_buf):
                T.ptx_mma(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    accum_dtype_abbrv,
                    A_local_buf.data,
                    A_offset,
                    B_local_buf.data,
                    B_offset,
                    C_local_buf.data,
                    C_offset,
                    T.bool(False),
                )

        else:
            scale_a_reg, scale_b_reg, scale_a_selector, scale_b_selector = scale_operands

            @T.macro
            def _atom_mma(A_local_buf, B_local_buf, C_local_buf):
                T.ptx_mma_scaled(
                    accum_dtype,
                    mma_prefix,
                    "row",
                    "col",
                    a_dtype_abbrv,
                    b_dtype_abbrv,
                    accum_dtype_abbrv,
                    A_local_buf.data,
                    A_offset,
                    B_local_buf.data,
                    B_offset,
                    C_local_buf.data,
                    C_offset,
                    T.bool(False),
                    scale_a_reg,
                    scale_b_reg,
                    scale_a_selector,
                    scale_b_selector,
                )

        return _atom_mma(A_local_buf, B_local_buf, C_local_buf)

    def make_mma_load_layout(self, local_buf: Buffer, matrix: str = "A", a_from_gemm_c: bool = False, is_regB: bool = False) -> T.Fragment:
        """PPU override: adds a_from_gemm_c parameter and PPU arch branches."""
        if a_from_gemm_c:
            return self.make_mma_load_layout_from_gemm_c(local_buf, matrix=matrix)
        if matrix == "B" and self.ppu_arch == 10 and is_regB:
            return self.make_mma_load_layout_ppu_b(local_buf)
        if matrix == "A" and self.ppu_arch == 10:
            dtype_bits = DataType(self.a_dtype).bits
            if dtype_bits in (16, 32):
                return self.make_mma_load_layout_ppu_a(local_buf)
        return self._make_mma_load_layout_default(local_buf, matrix=matrix)

    def make_mma_load_layout_ppu_a(self, local_buf: Buffer) -> T.Fragment:
        """PPU0010 A fragment layout for 16-bit and 32-bit MMA inputs."""
        assert is_fragment(local_buf), f"local_buf must be a fragment, but got {local_buf.scope()}"
        dtype_bits = DataType(self.a_dtype).bits
        if dtype_bits == 16:
            transform_func = ppu_shared_16x16_to_mma_32x8_layout_sr_a
            if self.a_transposed:
                transform_func = ppu_shared_16x16_to_mma_32x8_layout_trans_sr_a
        elif dtype_bits == 32:
            transform_func = ppu_shared_16x8_to_mma_32x4_layout_sr_a
        else:
            return self._make_mma_load_layout_default(local_buf, matrix="A")
        if self.a_transposed:
            base_transform_func = transform_func

            def transform_func(i, j):
                return base_transform_func(j, i)

        inverse_mma_load_layout = IndexMap.from_func(transform_func, index_dtype=T.int32)

        def forward_thread(i: int, j: int) -> int:
            lane_id, _ = inverse_mma_load_layout.map_indices([i, j])
            return lane_id

        def forward_index(i: int, j: int) -> int:
            _, local_id = inverse_mma_load_layout.map_indices([i, j])
            return local_id

        is_sr_axis_order = not self.a_transposed
        base_fragment = T.Fragment(
            [self.micro_size_x, self.micro_size_k] if is_sr_axis_order else [self.micro_size_k, self.micro_size_x],
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )
        warp_s, warp_r = self.warp_rows, self.chunk // self.micro_size_k
        if is_sr_axis_order:
            warp_fragment = base_fragment.repeat([warp_s, warp_r], repeat_on_thread=False, lower_dim_first=False)
            return warp_fragment.repeat([self.block_row_warps, 1], repeat_on_thread=True, lower_dim_first=True).replicate(
                self.block_col_warps
            )
        warp_fragment = base_fragment.repeat([warp_r, warp_s], repeat_on_thread=False, lower_dim_first=True)
        return warp_fragment.repeat([1, self.block_row_warps], repeat_on_thread=True, lower_dim_first=True).replicate(self.block_col_warps)

    def make_mma_load_layout_ppu_b(self, local_buf: Buffer) -> T.Fragment:
        assert is_fragment(local_buf), f"local_buf must be a fragment, but got {local_buf.scope()}"
        dtype_bits = DataType(self.b_dtype).bits
        if dtype_bits == 16:
            transform_func = shared_16x16_to_mma_32x8_layout_sr_b
            if not self.b_transposed:
                transform_func = ppu_shared_16x16_to_mma_32x8_layout_trans_sr_b
        else:
            return self._make_mma_load_layout_default(local_buf, matrix="B")

        is_sr_axis_order = self.b_transposed
        if not is_sr_axis_order:
            base_transform_func = transform_func

            def transform_func(i, j):
                return base_transform_func(j, i)

        inverse_mma_load_layout = IndexMap.from_func(transform_func, index_dtype=T.int32)

        def forward_thread(i: int, j: int) -> int:
            lane_id, _ = inverse_mma_load_layout.map_indices([i, j])
            return lane_id

        def forward_index(i: int, j: int) -> int:
            _, local_id = inverse_mma_load_layout.map_indices([i, j])
            return local_id

        micro_size_r, micro_size_s = self.micro_size_k, self.micro_size_y
        base_fragment = T.Fragment(
            [micro_size_s, micro_size_r] if is_sr_axis_order else [micro_size_r, micro_size_s],
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )
        warp_s, warp_r = self.warp_cols, self.chunk // self.micro_size_k
        block_s = self.block_col_warps
        replicate = self.block_row_warps
        if is_sr_axis_order:
            warp_fragment = base_fragment.repeat([warp_s, warp_r], repeat_on_thread=False, lower_dim_first=False)
            return warp_fragment.replicate(replicate).repeat([block_s, 1], repeat_on_thread=True, lower_dim_first=True)
        warp_fragment = base_fragment.repeat([warp_r, warp_s], repeat_on_thread=False, lower_dim_first=True)
        return warp_fragment.replicate(replicate).repeat([1, block_s], repeat_on_thread=True, lower_dim_first=True)

    def make_mma_load_layout_from_gemm_c(self, local_buf: Buffer, matrix: str = "A") -> T.Fragment:
        """RS GEMM: A fragment layout compatible with PPU MMA C-store layout."""
        if matrix != "A" or DataType(self.a_dtype).bits != 16:
            return self._make_mma_load_layout_default(local_buf, matrix=matrix)
        assert is_fragment(local_buf), f"local_buf must be a fragment, but got {local_buf.scope()}"
        if self.ppu_arch >= 15:
            return self._make_mma_load_layout_default(local_buf, matrix=matrix)
        inverse_mma_load_layout = IndexMap.from_func(ppu_shared_16x16_to_mma_32x8_layout_sr_a_from_gemm_c, index_dtype=T.int32)

        def forward_thread(i: int, j: int) -> int:
            lane_id, _ = inverse_mma_load_layout.map_indices([i, j])
            return lane_id

        def forward_index(i: int, j: int) -> int:
            _, local_id = inverse_mma_load_layout.map_indices([i, j])
            return local_id

        base_fragment = T.Fragment(
            [self.micro_size_x, self.micro_size_k],
            forward_thread_fn=forward_thread,
            forward_index_fn=forward_index,
        )
        warp_fragment = base_fragment.repeat(
            [self.warp_rows, self.chunk // self.micro_size_k],
            repeat_on_thread=False,
            lower_dim_first=False,
        )
        return warp_fragment.repeat(
            [self.block_row_warps, 1],
            repeat_on_thread=True,
            lower_dim_first=True,
        ).replicate(self.block_col_warps)

    def _make_mma_load_layout_default(self, local_buf: Buffer, matrix: Literal["A", "B"] = "A") -> T.Fragment:
        """Default load layout — delegates to base class (TensorCoreIntrinEmitter) make_mma_load_layout."""
        return TensorCoreIntrinEmitter.make_mma_load_layout(self, local_buf, matrix=matrix)
