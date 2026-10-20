/*!
 * \file tl/ppu/op/gemm_sp.cc
 * \brief PPU implementation registration for tl.gemm_sp instruction selection.
 */
#include "op/gemm_sp.h"
#include "op/gemm.h"
#include "support/check.h"

#include "op/builtin.h"
#include "op/utils.h"
#include "ppu/target_utils.h"

#include <cmath>
#include <cstdint>
#include <limits>
#include <utility>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace ppu {

namespace {

constexpr const char *kPpuMMASP = "ppu.mma.sp";

void FatalUnsupportedPpuArchGemmSP(const GemmSPNode &op, Target target,
                                   const char *inst_name) {
  LOG(FATAL) << inst_name << " requires an architecture newer than PPU's "
             << "supported ppu0010/ppu0015 targets. Got target=" << target
             << ", A(scope=" << op.A.scope() << ", dtype=" << op.A->dtype
             << "), B(scope=" << op.B.scope() << ", dtype=" << op.B->dtype
             << "), C(scope=" << op.C.scope() << ", dtype=" << op.C->dtype
             << "), M=" << op.M << ", N=" << op.N << ", K=" << op.K << ".";
}

std::pair<int, int>
ComputeDefaultWarpPartition(const GemmSPWarpPolicyNode &policy, int M, int N,
                            int num_warps, int k_n_per_warp) {
  int m_warp = 1, n_warp = 1;
  constexpr int kMPerWarp = 16;

  ICHECK(M % kMPerWarp == 0)
      << "M must be divisible by " << kMPerWarp << ", but got " << M;
  ICHECK(N % k_n_per_warp == 0)
      << "N must be divisible by " << k_n_per_warp << ", but got " << N;

  if (policy.IsFullRow()) {
    m_warp = num_warps;
    n_warp = 1;
    if (M % (m_warp * kMPerWarp) != 0) {
      int max_m_warps = M / kMPerWarp;
      m_warp = max_m_warps;
      n_warp = num_warps / m_warp;
      if (n_warp == 0)
        n_warp = 1;
    }
  } else if (policy.IsFullCol()) {
    m_warp = 1;
    n_warp = num_warps;
    if (N % (n_warp * k_n_per_warp) != 0) {
      int max_n_warps = N / k_n_per_warp;
      n_warp = max_n_warps;
      m_warp = num_warps / n_warp;
      if (m_warp == 0)
        m_warp = 1;
    }
  } else if (policy.IsSquare()) {
    int max_m_warps = M / kMPerWarp;
    float ideal_ratio = N > 0 ? static_cast<float>(M) / N : 1.0f;

    int best_m = 1;
    int best_n = 1;
    float best_balance = std::numeric_limits<float>::max();
    for (int m = 1; m <= max_m_warps && m <= num_warps; m++) {
      int n = num_warps / m;

      float m_per_warp = static_cast<float>(M) / (m * kMPerWarp);
      float n_per_warp = static_cast<float>(N) / (n * k_n_per_warp);
      if (m_per_warp < 1 || n_per_warp < 1)
        continue;
      if (m * n != num_warps)
        continue;

      float balance = std::abs(m_per_warp / n_per_warp - ideal_ratio);
      if (balance < best_balance) {
        best_balance = balance;
        best_m = m;
        best_n = n;
      }
    }

    m_warp = best_m;
    n_warp = best_n;
  } else {
    ICHECK(0) << "Unknown GemmSPWarpPolicy";
  }

  ICHECK(m_warp * n_warp == num_warps)
      << "m_warp * n_warp must equal num_warps, m_warp: " << m_warp
      << ", n_warp: " << n_warp << ", num_warps: " << num_warps;
  ICHECK(M % (m_warp * kMPerWarp) == 0 && N % (n_warp * k_n_per_warp) == 0)
      << "PPU T.gemm_sp warp partition would truncate output tile: M=" << M
      << ", N=" << N << ", m_warp=" << m_warp << ", n_warp=" << n_warp
      << ", num_warps=" << num_warps << ", policy="
      << (policy.IsFullRow()   ? "FullRow"
          : policy.IsFullCol() ? "FullCol"
                               : "Square")
      << ". M must be divisible by " << m_warp * kMPerWarp << " and N by "
      << n_warp * k_n_per_warp
      << "; adjust `threads`, the block tile shape, or the warp policy.";
  policy.m_warp = m_warp;
  policy.n_warp = n_warp;
  return {m_warp, n_warp};
}

} // namespace

struct GemmSP {
  static String SelectInst(const GemmSPNode &op, int block_size,
                           Target target) {
    if (op.isWgmma_) {
      FatalUnsupportedPpuArchGemmSP(op, target, "T.wgmma_gemm()");
    }
    if (op.isTcgen05_) {
      FatalUnsupportedPpuArchGemmSP(op, target, "T.tcgen05_gemm()");
    }
    return kPpuMMASP;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmSPWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    int num_warps = block_size / TargetPPUGetWarpSize(target);
    // PPU only targets ppu0010/ppu0015, so the warp partition uses the ppu0010+
    // rule.
    constexpr int kNPerWarp = 8;
    return ComputeDefaultWarpPartition(policy, M, N, num_warps, kNPerWarp);
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    return gemm_inst == kPpuMMASP;
  }

  static String InstructionKind(String gemm_inst) {
    if (gemm_inst == kPpuMMASP) {
      return "mma.sp";
    }
    return "unknown";
  }
};

} // namespace ppu

namespace {

bool MatchPpuGemmSPTarget(Target target) { return TargetIsPPU(target); }

bool RegisterPpuGemmSP() {
  RegisterGemmSPImpl(GemmSPImpl{
      "ppu.GemmSP",
      MatchPpuGemmSPTarget,
      ppu::GemmSP::SelectInst,
      ppu::GemmSP::ComputeWarpPartition,
      ppu::GemmSP::ReuseExistingSharedLayout,
  });
  return true;
}

const bool ppu_gemm_sp_registered = RegisterPpuGemmSP();

} // namespace

} // namespace tl
} // namespace tvm
