"""Access bounds and exact scale packs produced by the PPU scale loaders."""

import itertools

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.ppu.intrinsics.macro.mma_macro_generator import PPUTensorCoreIntrinEmitter
from tilelang.ppu.op.gemm.gemm_mma import _can_load_scale_fragment


PPU_TARGET = tvm.target.Target({"kind": "ppu", "arch": "ppu_15"})


def _scale_load_program(operand, warp_atoms, region_kind, is_m_first):
    row_warps, col_warps = (4, 2) if operand == "A" else (2, 4)
    threads = row_warps * col_warps * 32
    spatial = 4 * warp_atoms * 16
    packs = (warp_atoms + 3) // 4
    if region_kind == "vector":
        shape = (spatial,)
    elif region_kind == "matrix_slice":
        shape = (4, spatial + 16)
    else:
        shape = (2, 4, spatial + 16)
    emitter = PPUTensorCoreIntrinEmitter(
        a_dtype="float4_e2m1fn",
        b_dtype="float4_e2m1fn",
        accum_dtype="float32",
        block_row_warps=row_warps,
        block_col_warps=col_warps,
        warp_row_tiles=warp_atoms * 16 if operand == "A" else 64,
        warp_col_tiles=warp_atoms * 16 if operand == "B" else 64,
        chunk=128,
        is_m_first=is_m_first,
        ppu_arch=15,
    )

    @T.prim_func
    def program(Scales: T.Tensor(shape, "uint16"), Output: T.Tensor((threads, packs), "uint32")):
        T.func_attr({"target": PPU_TARGET})
        with T.Kernel(1, threads=threads):
            shared = T.alloc_shared(shape, "uint16")
            local = T.alloc_local((packs,), "uint32")
            T.copy(Scales, shared)
            if region_kind == "vector":
                emitter.load_scale(local, shared, 0, operand)
            elif region_kind == "matrix_slice":
                emitter.load_scale(local, shared[1:3, 8 : 8 + spatial], 1, operand)
            else:
                emitter.load_scale(local, shared[1:2, 1:3, 8 : 8 + spatial], 1, operand)
            tx = T.get_thread_binding()
            for pack in T.serial(packs):
                Output[tx, pack] = local[pack]

    return program


@pytest.mark.parametrize("operand", ["A", "B"])
@pytest.mark.parametrize("warp_atoms", [1, 2, 4, 8, 12])
@pytest.mark.parametrize("region_kind", ["vector", "matrix_slice", "pipeline_slice"])
@pytest.mark.parametrize("is_m_first", [False, True])
def test_scale_load_access_bounds(operand, warp_atoms, region_kind, is_m_first):
    program = _scale_load_program(operand, warp_atoms, region_kind, is_m_first)
    program = tilelang.transform.LetInline()(tvm.IRModule({"main": program}))["main"]
    calls = []
    domains = {}

    def visit(node):
        if isinstance(node, tvm.tirx.Call) and node.op.name == "tl.ptx_ldmatrix":
            calls.append(node)
        elif isinstance(node, tvm.tirx.For):
            domains[node.loop_var] = range(int(node.min), int(node.min) + int(node.extent))
        elif isinstance(node, tvm.tirx.AttrStmt) and node.attr_key == "thread_extent":
            domains[node.node.var] = range(int(node.value))

    tvm.tirx.stmt_functor.post_order_visit(program.body, visit)
    assert len(calls) == 1
    call = calls[0]
    assert int(call.args[0]) == 1
    assert int(call.args[1]) == 1
    src, dst = call.args[2:]
    assert src.op.name == dst.op.name == "tl.access_ptr"
    assert src.args[0].buffer.dtype == "uint16"
    assert dst.args[0].buffer.dtype == "uint32"
    assert int(src.args[1]) == 8  # One row supplied to the collective load.
    assert int(dst.args[1]) == 1  # One packed register written per thread.
    assert int(src.args[2]) == 1
    assert int(dst.args[2]) == 2

    analyzer = tvm.arith.Analyzer()
    for ptr in (src, dst):
        load = ptr.args[0]
        variables = set()
        for index in load.indices:
            tvm.tirx.stmt_functor.post_order_visit(
                index, lambda node, found=variables: found.add(node) if isinstance(node, tvm.tirx.Var) else None
            )
        variables = list(variables)
        for values in itertools.product(*(domains[var] for var in variables)):
            subst = {var: tvm.tirx.IntImm(var.dtype, value) for var, value in zip(variables, values)}
            bounds = [analyzer.const_int_bound(tvm.tirx.stmt_functor.substitute(index, subst)) for index in load.indices]
            for bound, extent in zip(bounds, load.buffer.shape):
                assert 0 <= bound.min_value <= bound.max_value < int(extent)
            assert bounds[-1].max_value + int(ptr.args[1]) <= int(load.buffer.shape[-1])
            if ptr.same_as(src) and region_kind != "vector":
                assert bounds[-2].min_value == bounds[-2].max_value == 2
                assert bounds[-1].min_value >= 8
                assert bounds[-1].max_value + 8 <= 8 + 4 * warp_atoms * 16
                if region_kind == "pipeline_slice":
                    assert bounds[0].min_value == bounds[0].max_value == 1


@tilelang.testing.requires_ppu_compute_version_ge(1, 5)
@pytest.mark.parametrize("operand", ["A", "B"])
@pytest.mark.parametrize("warp_atoms", [1, 2, 4, 8, 12])
@pytest.mark.parametrize("region_kind", ["vector", "matrix_slice", "pipeline_slice"])
@pytest.mark.parametrize("is_m_first", [False, True])
def test_scale_load_packed_values(operand, warp_atoms, region_kind, is_m_first):
    program = _scale_load_program(operand, warp_atoms, region_kind, is_m_first)
    shape = tuple(int(dim) for dim in program.buffer_map[program.params[0]].shape)
    generator = torch.Generator().manual_seed(17)
    scales = torch.randint(0, 65536, shape, generator=generator, dtype=torch.int64)
    kernel = tilelang.compile(program, target=PPU_TARGET, out_idx=[1])
    result = kernel(scales.to(torch.uint16).cuda()).cpu().to(torch.int64)
    if region_kind == "vector":
        selected = scales
    elif region_kind == "matrix_slice":
        selected = scales[2, 8 : 8 + 4 * warp_atoms * 16]
    else:
        selected = scales[1, 2, 8 : 8 + 4 * warp_atoms * 16]

    row_warps, col_warps = (4, 2) if operand == "A" else (2, 4)
    expected = torch.zeros_like(result)
    consumed = torch.zeros_like(result, dtype=torch.bool)
    for warp_m, warp_n in itertools.product(range(row_warps), range(col_warps)):
        warp = warp_m * col_warps + warp_n if is_m_first else warp_n * row_warps + warp_m
        spatial_warp = warp_m if operand == "A" else warp_n
        for atom, row in itertools.product(range(warp_atoms), range(8)):
            # Each MMA atom consumes two uint16 row/column scales per lane.
            # Its selector chooses one of four lane groups in the scale pack.
            lane = row * 4 + atom % 4
            index = (spatial_warp * warp_atoms + atom) * 16 + row
            expected[warp * 32 + lane, atom // 4] = selected[index] | (selected[index + 8] << 16)
            consumed[warp * 32 + lane, atom // 4] = True
    torch.testing.assert_close(result[consumed], expected[consumed], rtol=0, atol=0)


@pytest.mark.parametrize(
    "strides,elem_offset,alignment,expected",
    [
        ((72, 1), 8, 64, True),
        ((65, 1), 0, 64, False),
        ((128, 2), 0, 64, False),
        ((64, 1), 1, 64, False),
        ((64, 1), 0, 8, False),
    ],
)
def test_scale_fragment_physical_address(strides, elem_offset, alignment, expected):
    buffer = tvm.tirx.decl_buffer((2, 64), "uint16", scope="shared", strides=strides, elem_offset=elem_offset, data_alignment=alignment)
    region = tvm.tirx.BufferRegion(buffer, [tvm.ir.Range(0, 2), tvm.ir.Range(0, 64)])
    assert _can_load_scale_fragment(region, {}, 2) == expected


def test_scale_fragment_symbolic_prefix_and_layout():
    stage = tvm.tirx.Var("stage", "int32")
    buffer = tvm.tirx.decl_buffer((2, 2, 80), "uint16", scope="shared")
    region = tvm.tirx.BufferRegion(buffer, [tvm.ir.Range(stage, 1), tvm.ir.Range(0, 2), tvm.ir.Range(8, 64)])
    assert _can_load_scale_fragment(region, {}, 2)
    unknown_offset = tvm.tirx.BufferRegion(buffer, [tvm.ir.Range(0, 1), tvm.ir.Range(0, 2), tvm.ir.Range(stage, 64)])
    assert not _can_load_scale_fragment(unknown_offset, {}, 2)
    layout = tilelang.Layout(buffer.shape, lambda p, k, i: (p, k, i ^ 1))
    assert not _can_load_scale_fragment(region, {buffer: layout}, 2)


@pytest.mark.parametrize(
    "case,operand",
    [
        ("aligned", "A"),
        ("aligned_slice", "A"),
        ("global", "A"),
        ("global", "B"),
        ("offset", "A"),
        ("offset", "B"),
        ("row_stride", "A"),
        ("row_stride", "B"),
    ],
)
def test_scale_fragment_selection(case, operand):
    k_tiles = 1 if case == "offset" else 2
    K = 64 * k_tiles
    offset = 8 if case == "aligned_slice" else int(case == "offset")
    width = 80 if case == "aligned_slice" else 65 if case in ("offset", "row_stride") else 64
    a_width, b_width = (width, 64) if operand == "A" else (64, width)
    a_offset, b_offset = (offset, 0) if operand == "A" else (0, offset)
    global_a = case == "global" and operand == "A"
    global_b = case == "global" and operand == "B"

    @T.prim_func
    def program(
        A: T.Tensor((64, K), "float4_e2m1fn"),
        B: T.Tensor((64, K), "float4_e2m1fn"),
        SFA: T.Tensor((k_tiles, a_width), "uint16"),
        SFB: T.Tensor((k_tiles, b_width), "uint16"),
        C: T.Tensor((64, 64), "float32"),
    ):
        T.func_attr({"target": PPU_TARGET})
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, K), "float4_e2m1fn")
            B_shared = T.alloc_shared((64, K), "float4_e2m1fn")
            SFA_shared = T.alloc_shared((k_tiles, a_width), "uint16")
            SFB_shared = T.alloc_shared((k_tiles, b_width), "uint16")
            C_local = T.alloc_fragment((64, 64), "float32")
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.copy(SFA, SFA_shared)
            T.copy(SFB, SFB_shared)
            T.clear(C_local)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                transpose_B=True,
                scale_A=SFA[:, a_offset : a_offset + 64] if global_a else SFA_shared[:, a_offset : a_offset + 64],
                scale_B=SFB[:, b_offset : b_offset + 64] if global_b else SFB_shared[:, b_offset : b_offset + 64],
            )
            T.copy(C_local, C)

    with PPU_TARGET, tvm.transform.PassContext(opt_level=3):
        source = tilelang.lower(program, target=PPU_TARGET).kernel_source
    assert "tl::mma_sync_scaled<" in source
    expected_loads = 2 if case in ("aligned", "aligned_slice") else 0
    assert source.count("tl::tix_ldmatrix_x1_trans(") == expected_loads
    if not expected_loads:
        mma_source = source[source.index("tl::mma_sync_scaled<") :]
        assert ("SFA[" if global_a else "SFA_shared)[") in mma_source
        assert ("SFB[" if global_b else "SFB_shared)[") in mma_source


@pytest.mark.parametrize("path", ["SR", "RS", "RR"])
def test_mxfp4_register_inputs_lowering(path):
    """Check emitted instructions directly; FP4 fragment layout inference is not supported."""
    tir = tvm.tirx
    a = tir.decl_buffer((64, 128), "float4_e2m1fn", name="A", scope="local.fragment" if path[0] == "R" else "shared")
    b = tir.decl_buffer((64, 128), "float4_e2m1fn", name="B", scope="local.fragment" if path[1] == "R" else "shared")
    c = tir.decl_buffer((64, 64), "float32", name="C", scope="local.fragment")
    scales = [tir.decl_buffer((2, 64), "uint16", name=name, scope="shared") for name in ("SFA", "SFB")]
    call = T.gemm(a, b, c, transpose_B=True, scale_A=scales[0], scale_B=scales[1])
    gemm = call.op.get_attr("TLOpBuilder")(call.args, call.annotations)
    with PPU_TARGET:
        program = gemm.lower({}, PPU_TARGET, tvm.ir.Range(0, 128), tir.Var("tx", "int32"), tir.IntImm("int32", 0))

    loads, mmas, loops = [], [], []

    def collect(node):
        if isinstance(node, tir.For):
            loops.append(node)
        elif isinstance(node, tir.Call):
            if node.op.name == "tl.ptx_ldmatrix" and int(node.args[0]) == 1:
                loads.append(node)
            elif node.op.name == "tirx.ptx_mma":
                mmas.append(node)

    tir.stmt_functor.post_order_visit(program.body, collect)
    assert len(loads) == 2
    assert len(mmas) == 1
    mma = mmas[0]
    analyzer = tvm.arith.Analyzer()
    scale_loads = {load.args[2].args[0].buffer.name: load for load in loads}
    assert set(scale_loads) == {"SFA", "SFB"}
    k_index = scale_loads["SFA"].args[2].args[0].indices[0]
    k_loop = next(loop for loop in loops if loop.loop_var.same_as(k_index))
    assert int(k_loop.min) == 0 and int(k_loop.extent) == 2
    body_calls = []
    tir.stmt_functor.post_order_visit(k_loop.body, lambda node: body_calls.append(node) if isinstance(node, tir.Call) else None)
    assert all(any(node.same_as(call) for node in body_calls) for call in [*loads, mma])

    def evaluate(expr, substitutions):
        def fold(node):
            if isinstance(node, tir.IntImm):
                return int(node)
            if isinstance(node, tir.Mul):
                return fold(node.a) * fold(node.b)
            if isinstance(node, tir.Call) and node.op.name == "tirx.bitwise_xor":
                return fold(node.args[0]) ^ fold(node.args[1])
            raise AssertionError(f"Expected a constant scale address, got {node}")

        substitutions = {var: tir.IntImm(var.dtype, value) for var, value in substitutions.items()}
        return fold(analyzer.simplify(tir.stmt_functor.substitute(expr, substitutions)))

    # Distinct packed E8M0 pairs identify A/B, row/column, and K64.
    spatial = torch.arange(64)[None, :]
    k64 = torch.arange(2)[:, None]
    for operand, name in enumerate(("SFA", "SFB")):
        load = scale_loads[name]
        src, dst = load.args[2:]
        assert int(load.args[1]) == 1
        assert int(src.args[1]) == 8 and int(dst.args[1]) == 1
        source = src.args[0]
        assert source.buffer.dtype == "uint16"
        assert dst.args[0].buffer.dtype == "uint32"
        assert analyzer.can_prove(source.indices[0] == k_index)
        variables = set()
        tir.stmt_functor.post_order_visit(
            source.indices[1], lambda node, found=variables: found.add(node) if isinstance(node, tir.Var) else None
        )
        assert len(variables) == 1
        tx = variables.pop()
        packed = (96 + spatial + 5 * k64 + operand) | ((192 + spatial // 2 + 7 * k64 + operand) << 8)
        for ki, thread in itertools.product(range(2), range(128)):
            row, atom = (thread % 32) // 4, thread % 4
            if atom >= 2:
                continue
            # Transposed x1 places two adjacent matrix rows into each lane's
            # uint32. Row addresses come from the first eight lanes.
            supplied = []
            for matrix_row in (2 * atom, 2 * atom + 1):
                substitutions = {k_index: ki, tx: thread // 32 * 32 + matrix_row}
                indices = [evaluate(index, substitutions) for index in source.indices]
                supplied.append(int(packed[indices[0], indices[1] + row]))
            actual = supplied[0] | (supplied[1] << 16)
            spatial_warp = thread // 32 % 2 if operand == 0 else thread // 64
            spatial_index = spatial_warp * 32 + atom * 16 + row
            expected = int(packed[ki, spatial_index]) | (int(packed[ki, spatial_index + 8]) << 16)
            assert actual == expected
        scale_arg = mma.args[13 + operand]
        assert isinstance(scale_arg, tir.BufferLoad)
        assert scale_arg.buffer.same_as(dst.args[0].buffer)
        assert analyzer.can_prove(scale_arg.indices[0] == 0)

    # Register inputs advance to their second K64 fragment; shared inputs
    # reload the same local scratch buffer on each iteration.
    assert len(mma.args) == 17
    for operand, offset_arg in enumerate((7, 9)):
        offset = mma.args[offset_arg]
        delta = tir.stmt_functor.substitute(offset, {k_index: tir.IntImm("int32", 1)}) - tir.stmt_functor.substitute(
            offset, {k_index: tir.IntImm("int32", 0)}
        )
        assert analyzer.can_prove(delta == (64 if path[operand] == "R" else 0))
    assert analyzer.can_prove(mma.args[7] == mma.args[15] * 32 + (k_index * 64 if path[0] == "R" else 0))
    assert analyzer.can_prove(mma.args[9] == mma.args[16] * 32 + (k_index * 64 if path[1] == "R" else 0))


if __name__ == "__main__":
    tilelang.testing.main()
