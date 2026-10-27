"""PPU-specific transformation frontends."""

from .. import _ffi_api


def LowerThreadAllreduce():
    """Lower thread allreduce with PPU-safe subword synchronization."""
    return _ffi_api.LowerThreadAllreduce()  # type: ignore


def LowerL2Persistent():
    """LowerL2Persistent."""
    # PPU: call the independently registered PPU transform implementation.
    return _ffi_api.LowerL2Persistent()  # type: ignore


def LowerL2PersistentCache():
    """LowerL2PersistentCache.

    Materializes L2 persistent cache access-policy-window calls from the
    ``l2_persistent_map`` PrimFunc attribute set by ``LowerL2Persistent``.
    """
    # PPU: call the independently registered PPU transform implementation.
    if hasattr(_ffi_api, "LowerL2PersistentCache"):
        return _ffi_api.LowerL2PersistentCache()  # type: ignore
    return lambda f: f


def LowerLDGSTG():
    """Lower Ramp-based global memory load/store to ldg/stg intrinsics."""
    # PPU: call the independently registered PPU transform implementation.
    return _ffi_api.LowerLDGSTG()  # type: ignore


def PackSubwordWarpShuffle():
    """Fuse proven scalar FP8/INT8 relayouts into 32-bit warp shuffles."""
    return _ffi_api.PackSubwordWarpShuffle()  # type: ignore


def MarkPpuSyncCalls(have_pdl: bool = False):
    """MarkPpuSyncCalls."""
    # PPU: call the independently registered PPU transform implementation.
    return _ffi_api.MarkPpuSyncCalls(have_pdl)  # type: ignore


def InjectPpuUniform():
    """Wrap PPU swizzled ldmatrix thread-index address arguments as uniform."""
    # PPU: call the independently registered PPU transform implementation.
    return _ffi_api.InjectPpuUniform()  # type: ignore


def InjectAIUSyncBarrier():
    """Inject commit/wait sync barriers for AIU loads in stage=0 loops."""
    # PPU: call the independently registered PPU transform implementation.
    return _ffi_api.InjectAIUSyncBarrier()  # type: ignore


def AnnotateChainedGemm():
    """Annotate chained gemm (A from prior GEMM C) with 'a_from_gemm_c'."""
    return _ffi_api.AnnotateChainedGemm()  # type: ignore


def LayoutInference():
    """PPU-specific layout inference that handles GEMM RS SRCA layout conflicts."""
    # PPU: call the independently registered PPU transform implementation.
    return _ffi_api.LayoutInference()  # type: ignore


def LowerTileOp():
    """PPU-specific LowerTileOp with AIU/swizzle ldmatrix support."""
    # PPU: call the independently registered PPU transform implementation.
    return _ffi_api.LowerTileOp()  # type: ignore


def ReorderAIULoads():
    """Reorder aiu_load instructions by their dst buffer's first-use position."""
    # PPU: call the independently registered PPU transform implementation.
    return _ffi_api.ReorderAIULoads()  # type: ignore


def PersistThreadblock():
    """PersistThreadblock."""
    # PPU: call the independently registered PPU transform implementation.
    return _ffi_api.PersistThreadblock()  # type: ignore


__all__ = [
    "InjectAIUSyncBarrier",
    "InjectPpuUniform",
    "LowerL2PersistentCache",
    "LowerLDGSTG",
    "LowerThreadAllreduce",
    "LowerL2Persistent",
    "LowerTileOp",
    "AnnotateChainedGemm",
    "PackSubwordWarpShuffle",
    "MarkPpuSyncCalls",
    "PersistThreadblock",
    "ReorderAIULoads",
]
