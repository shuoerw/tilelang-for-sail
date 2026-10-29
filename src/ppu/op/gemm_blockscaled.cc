/*!
 * \file tl/ppu/op/gemm_blockscaled.cc
 * \brief PPU instruction selection for block-scaled GEMM.
 */

#include "op/gemm_blockscaled.h"

#include "ppu/target_utils.h"

namespace tvm {
namespace tl {
namespace ppu {

namespace {

constexpr const char *kPpuMMABlockScaled = "ppu.mma.blockscaled";

ffi::String SelectBlockScaledGemmInst(const GemmBlockScaled & /*op*/,
                                      int /*block_size*/,
                                      const Target & /*target*/) {
  return kPpuMMABlockScaled;
}

bool MatchPpuBlockScaledGemmTarget(Target target) {
  return TargetIsPPU(target);
}

bool RegisterPpuGemmBlockScaled() {
  RegisterGemmBlockScaledImpl(GemmBlockScaledImpl{
      "ppu.GemmBlockScaled",
      MatchPpuBlockScaledGemmTarget,
      SelectBlockScaledGemmInst,
  });
  return true;
}

const bool ppu_gemm_blockscaled_registered = RegisterPpuGemmBlockScaled();

} // namespace

} // namespace ppu
} // namespace tl
} // namespace tvm
