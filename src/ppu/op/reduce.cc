/*!
 * \file tl/ppu/op/reduce.cc
 * \brief PPU implementation registration for tl.reduce AllReduce lowering.
 */

#include "backend/common/op/reduce.h"

#include "backend/common/target_utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace ppu {

struct Reduce : backend::ReduceLowerer<Reduce> {
  static bool AllReduceNeedsWorkspace(int reducing_threads, int, Target) {
    return reducing_threads > 32;
  }

  static bool SupportsFp16Bf16NanReduce(Target target) {
    return TargetIsPPU(target);
  }

  static int GetPreferredVectorizedSize(const ReduceOpNode &op, Target target) {
    if (!TargetIsPPU(target)) {
      return 1;
    }
    return backend::reduce::GetPreferredVectorizedSize(op.dst->dtype);
  }

  static std::string MakeBatchAllReduce(std::string reducer,
                                        int reducing_threads, int scale,
                                        PrimExpr thread_offset,
                                        PrimExpr all_threads, int batch,
                                        int workspace_stride, Target target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset;
    ss << ", tl::SyncThreadsBarrier";
    ss << ", " << batch << ", " << workspace_stride << ">::run_batch";
    return ss.str();
  }

  static std::string MakeScalarAllReduce(std::string reducer,
                                         int reducing_threads, int scale,
                                         PrimExpr thread_offset,
                                         PrimExpr all_threads, Target target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset;
    ss << ">::run";
    return ss.str();
  }
};

} // namespace ppu

namespace {

bool MatchPpuReduceTarget(Target target) { return TargetIsPPU(target); }

bool RegisterPpuReduce() {
  RegisterReduceImpl(ReduceImpl{
      "ppu.Reduce",
      MatchPpuReduceTarget,
      ppu::Reduce::Lower,
  });
  return true;
}

const bool ppu_reduce_registered = RegisterPpuReduce();

} // namespace

} // namespace tl
} // namespace tvm
