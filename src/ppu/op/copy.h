/*!
 * \file tl/ppu/op/copy.h
 * \brief PPU copy instruction classification helpers.
 */

#ifndef TVM_TL_BACKEND_PPU_OP_COPY_H_
#define TVM_TL_BACKEND_PPU_OP_COPY_H_

#include "ppu/op/builtin.h"
#include "op/copy.h"
#include "support/check.h"

#include <cstddef>
#include <cstdint>
#include <string>

namespace tvm {
namespace tl {
namespace ppu {

using namespace tirx;
using namespace ffi;

enum class CopyInst : uint8_t {
  kNormal = 0,
  kLDSM = 1,
  kCPAsync = 3,
  kAiuLoad = 4, // PPU AIU global->swizzled-shared bulk copy
  kInvalid = 255,
};

const char *CopyInstToString(CopyInst inst);
bool CopyInstIsCPAsync(CopyInst inst);

// Whether an AIU load relies on compiler-inserted commit/wait operations.
inline bool IsCompilerSyncAiuLoad(const CallNode *call) {
  if (!call->op.same_as(tl::ppu_aiu_load())) {
    return false;
  }
  if (auto flag = call->annotations.Get(attr::kAsyncCopyNoImplicitCommitWait)) {
    const auto *value = flag->as<IntImmNode>();
    ICHECK(value) << "no_implicit_async_commit_wait must be a constant boolean";
    return value->value == 0;
  }
  return true;
}

struct CopyAnalysisContext {
  Target target;
  const LayoutMap *layout_map = nullptr;
  arith::Analyzer *analyzer = nullptr;
  bool emit_diagnostics = false;
};

struct CopyInstSelection {
  CopyInst inst = CopyInst::kNormal;
  bool supported = true;
  std::string reason;
};

// Final PPU lowering decision. Explicit T.tma_copy/T.async_copy semantics are
// enforced here and reported through CopyInstSelection::reason.
CopyInstSelection SelectCopyInstForLowering(const CopyNode &op,
                                            const CopyAnalysisContext &ctx);

} // namespace ppu
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_PPU_OP_COPY_H_
