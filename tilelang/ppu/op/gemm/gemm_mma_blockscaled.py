"""PPU block-scaled MMA lowering."""

from tilelang.tileop.gemm_blockscaled.gemm_blockscaled_base import GemmBlockScaledMixin

from .gemm_mma import PPUGemmMMA

GEMM_INST_MMA_BLOCK_SCALED_PPU = "ppu.mma.blockscaled"


class PPUGemmMMABlockScaled(GemmBlockScaledMixin, PPUGemmMMA):
    @property
    def is_blockscaled(self) -> bool:
        return True
