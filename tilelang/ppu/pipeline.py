from __future__ import annotations

from tvm import IRModule, ir, s_tir, tirx
from tvm.target import Target
from tvm.tirx import PrimFunc, SBlock
from tvm.tirx.stmt_functor import post_order_visit

import tilelang
from tilelang.backend.pass_pipeline.pipeline import PassPipeline, register_pipeline
from tilelang.backend.pass_pipeline.pipeline_utils import (
    LayoutVisual,
    allow_vectorize,
    should_disable_shared_memory_reuse,
    should_enable_aggressive_merge,
    should_enable_race_check,
    should_force_let_inline,
)
from tilelang.contrib.hgcc import (
    get_target_compute_version,
    have_mbarrier,
    have_pdl,
)


def _module_has_shared_barrier(mod: IRModule) -> bool:
    """Whether any function allocates a shared.barrier / shared.cluster_barrier
    buffer (i.e. uses ``T.alloc_barrier``).
    """
    found = False

    def visit(node):
        nonlocal found
        if isinstance(node, SBlock):
            for buffer in node.alloc_buffers:
                if buffer.scope() in ("shared.barrier", "shared.cluster_barrier"):
                    found = True

    for _, func in mod.functions.items():
        if isinstance(func, PrimFunc):
            post_order_visit(func.body, visit)
    return found


@tirx.transform.prim_func_pass(opt_level=0)
def _AlignAIUSharedMemory(func, mod, ctx):
    """Use uniform 128-byte shared alignment in kernels that issue AIU loads."""
    has_aiu = False
    buffers = []

    def visit(node):
        nonlocal has_aiu
        if isinstance(node, tirx.Call) and node.op == ir.Op.get("tl.ppu_aiu_load"):
            has_aiu = True
        elif isinstance(node, tirx.AllocBuffer) and node.buffer.scope() in ("shared", "shared.dyn"):
            buffers.append(node.buffer)

    post_order_visit(func.body, visit)
    if not has_aiu:
        return func
    alignments = dict(func.attrs.get("tl.smem_alignment_map", {})) if func.attrs else {}
    for buffer in buffers:
        name = buffer.data.name
        alignments[name] = tirx.IntImm("int32", max(128, int(alignments.get(name, 0))))
    return func.with_attr("tl.smem_alignment_map", alignments)


def PPUPassPipelineBodyPrologue(mod: IRModule, target: Target) -> IRModule:
    mod = tirx.transform.BindTarget(target)(mod)
    mod = tilelang.transform.MaterializeKernelLaunch()(mod)
    if should_force_let_inline():
        # Force-let inline whenever the pass config requests it.
        mod = tilelang.transform.LetInline()(mod)
    # Add wrapper for single buf store
    mod = tilelang.transform.AddWrapperForSingleBufStore()(mod)
    # Normalize negative indices to canonical non-negative form
    mod = tilelang.transform.LegalizeNegativeIndex()(mod)
    # Verify parallel loop correctness
    if should_enable_race_check():
        mod = tilelang.transform.VerifyParallelLoop()(mod)
    # Inject assumes to speedup tvm prover
    mod = tilelang.transform.InjectAssumes()(mod)
    # Simplify the IR expressions
    mod = tilelang.transform.Simplify()(mod)
    # Set layouts for reducers
    mod = tilelang.transform.LayoutReducer()(mod)

    # Normalize if-without-else wrappers before pipeline planning. This keeps
    # pipeline body extraction focused on canonical SeqStmt bodies.
    mod = tilelang.transform.IfStmtBinding()(mod)

    # Run pipeline planning and software-pipeline rewriting before layout
    # inference so inferred layouts see the final pipelined structure directly.
    mod = tilelang.transform.PipelinePlanning()(mod)
    mod = tilelang.transform.InjectSoftwarePipeline()(mod)
    mod = tilelang.transform.Simplify()(mod)

    # @PPU-specific: Annotate chained gemm (A from prior GEMM C).
    # Must run after software pipelining and before LayoutInference.
    mod = tilelang.ppu.transform.AnnotateChainedGemm()(mod)

    # @PPU-specific: Use PPU LayoutInference that handles GEMM RS SRCA
    # layout conflicts via trans_buffer mechanism
    mod = tilelang.ppu.transform.LayoutInference()(mod)
    # Visualize the layout
    LayoutVisual(mod)
    # Lower high-level tile operations to low-level operations
    mod = tilelang.ppu.transform.LowerTileOp()(mod)

    # @PPU specific
    # Lower l2 persistent map
    mod = tilelang.ppu.transform.LowerL2Persistent()(mod)
    # Decouple type cast vectorization constraints before vectorization
    mod = tilelang.transform.DecoupleTypeCast()(mod)
    # Legalize vectorized loops to ensure they are valid
    mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    # Add safety checks for memory accesses
    mod = tilelang.transform.LegalizeSafeMemoryAccess()(mod)
    # Lower frontend pointer metadata op to standard tvm_access_ptr
    mod = tilelang.transform.LowerAccessPtr()(mod)
    # Simplify again to clean up any duplicated conditions
    # that may have been introduced by safety checks
    # use an enhanced pass to simplify the dynamic symbolics
    # TODO(lei): return to tir pass when kSymbolicBound simplification
    # is merged into tvm.
    mod = tilelang.transform.Simplify()(mod)

    # @PPU-specific
    # Reorder AIU loads by dst buffer first-use position (stage=0 only)
    mod = tilelang.ppu.transform.ReorderAIULoads()(mod)
    # Inject commit/wait sync barriers for AIU loads in stage=0 loops
    mod = tilelang.ppu.transform.InjectAIUSyncBarrier()(mod)

    # Hoist any root-block annotations to PrimFunc attrs if pass is available
    mod = tilelang.transform.HoistNonRestrictParams()(mod)
    return mod


def PPUPassPipelineBody(mod: IRModule, target: Target) -> IRModule:
    pass_ctx = tilelang.transform.get_pass_context()

    mod = PPUPassPipelineBodyPrologue(mod, target)

    # Pipeline barriers are now created at final expanded size by
    # InjectSoftwarePipeline, so no late MVB barrier fixup is needed.
    # Buffer allocation placement is handled uniformly for both paths.
    mod = tilelang.transform.PlanAndUpdateBufferAllocationLocation()(mod)
    # @PPU-specific
    # LowerSharedBarrier emits hardware mbarrier code (the cutlass Barrier type,
    # tl::tl_shuffle_elect, tl::fence_barrier_init) which only exists on ppu0015+.
    # Reject T.alloc_barrier() up front on pre-PPU1.5 targets instead of letting
    # hgcc fail later with cryptic "identifier 'Barrier' is undefined" errors.
    if not have_mbarrier(target) and _module_has_shared_barrier(mod):
        compute_version = get_target_compute_version(target)
        raise ValueError(
            f"T.alloc_barrier() requires ppu0015 (PPU 1.5) or later, but the current "
            f"target is ppu arch {compute_version.replace('.', '')} (PPU arch version "
            f"{compute_version}). Hardware mbarrier operations (Barrier type, "
            f"tl_shuffle_elect, fence_barrier_init) are not available on this "
            f"architecture. Use __syncthreads() or named barriers for pre-PPU1.5 "
            f"targets."
        )

    mod = tilelang.transform.HoistGlobalBufferAllocations()(mod)
    mod = tilelang.transform.LowerOpaqueBlock()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tirx.transform.NarrowDataType(32)(mod)
    mod = tilelang.transform.FlattenBuffer()(mod)
    # ConfigIndexBitwidth must be applied after FlattenBuffer
    # as it will flatten index computing
    mod = tilelang.transform.ConfigIndexBitwidth()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tilelang.transform.VectorizeLoop(enable_vectorize=allow_vectorize(pass_ctx=pass_ctx))(mod)
    mod = tilelang.transform.StorageRewrite()(mod)
    mod = tilelang.transform.LoopUnswitching()(mod)
    mod = tilelang.transform.UnrollLoop()(mod)
    mod = s_tir.transform.RenormalizeSplitPattern()(mod)
    mod = tirx.transform.Simplify()(mod)
    # @PPU-specific: combine proven scalar subword relayouts into packed
    # uint32 warp shuffles.
    mod = tilelang.ppu.transform.PackSubwordWarpShuffle()(mod)
    mod = tirx.transform.Simplify()(mod)
    mod = tirx.transform.RemoveNoOp()(mod)
    mod = s_tir.transform.HoistIfThenElse()(mod)

    mod = tirx.transform.VerifyMemory()(mod)
    mod = tirx.transform.AnnotateEntryFunc()(mod)
    # TODO(lei): This is a hack to make sure the
    # thread level allreduce pass can be applied
    # in TL. As Tl only use one thread dimension
    # the var binding information will be lost
    # in the lowering process with Legalization
    # and Simplify pass.
    # We can find a way better to create var instead
    # of putting the LowerThreadAllreduce before
    # the Legalization.
    mod = s_tir.transform.InferFragment()(mod)
    mod = tilelang.transform.LowerThreadAllreduce()(mod)

    # @PPU-specific
    mod = tilelang.ppu.transform.LowerLDGSTG()(mod)
    # @PPU-specific
    # LowerL2PersistentCache materializes L2 persistent cache
    # access-policy-window calls from the ``l2_persistent_map`` PrimFunc
    # attribute set by LowerL2Persistent.
    mod = tilelang.ppu.transform.LowerL2PersistentCache()(mod)

    mod = tilelang.transform.AnnotateDeviceRegions()(mod)
    mod = tilelang.transform.SplitHostDevice()(mod)

    # @PPU-specific
    # Mark the function contains pdl_sync or pdl_trigger
    mod = tilelang.ppu.transform.MarkPpuSyncCalls(have_pdl(target))(mod)
    mod = tilelang.transform.AnnotateReadOnlyParams()(mod)

    # MergeSharedMemoryAllocations must be applied after SplitHostDevice
    # because the merged allocation site is at the beginning of each device function
    enable_aggressive_merge = should_enable_aggressive_merge(pass_ctx=pass_ctx, target=target)
    disable_reuse = should_disable_shared_memory_reuse(pass_ctx=pass_ctx)
    mod = _AlignAIUSharedMemory(mod)
    mod = tilelang.transform.MergeSharedMemoryAllocations(enable_aggressive_merge=enable_aggressive_merge, disable_reuse=disable_reuse)(mod)

    mod = tilelang.transform.ThreadSync("shared")(mod)
    mod = tilelang.transform.ThreadSync("shared.dyn")(mod)

    mod = tilelang.transform.MergeIfStmt()(mod)

    # @PPU-specific
    # Rewrite warp-aligned if-guards into scalar PPU uniform branches.
    mod = tilelang.ppu.transform.InjectPpuUniform()(mod)

    mod = tilelang.transform.MakePackedAPI()(mod)
    mod = tilelang.transform.Simplify()(mod)
    mod = tilelang.transform.LowerDeviceKernelLaunch()(mod)

    # @PPU-specific
    # Transform threadblock to persistent threadblock
    mod = tilelang.ppu.transform.PersistThreadblock()(mod)

    return mod


ppu_pipeline = PassPipeline("ppu", PPUPassPipelineBody)

register_pipeline(ppu_pipeline)
