/*!
 * \file tl/ppu/op/copy_analysis.cc
 * \brief PPU copy instruction classification helpers.
 */

#include "ppu/op/copy.h"
#include "support/check.h"
#include <tvm/runtime/logging.h>

#include "op/builtin.h"
#include "op/utils.h"
#include "ppu/target_utils.h"

#include <tvm/tirx/transform.h>

#include <optional>
#include <sstream>
#include <utility>

#include <algorithm>
#include <climits>

namespace tvm {
namespace tl {
namespace ppu {

using namespace tirx;
using namespace ffi;

namespace {

bool GetBoolAnnotation(const CopyNode &op, const char *key) {
  if (auto val = op.annotations.Get(key)) {
    if (auto int_val = val->as<IntImmNode>()) {
      return int_val->value != 0;
    }
  }
  return false;
}

bool GetIsTmaCopy(const CopyNode &op) {
  return GetBoolAnnotation(op, "is_tma_copy");
}

int64_t GetClusterMask(const CopyNode &op) {
  if (auto val = op.annotations.Get("cluster_mask")) {
    if (auto int_val = val->as<IntImmNode>()) {
      return int_val->value;
    }
  }
  return 0;
}

bool GetIsAsyncCopy(const CopyNode &op) {
  if (GetBoolAnnotation(op, "is_async_copy")) {
    return true;
  }
  return GetBoolAnnotation(op, "force_cp_async");
}

bool GetNoImplicitAsyncCommitWait(const CopyNode &op) {
  return GetBoolAnnotation(op, attr::kAsyncCopyNoImplicitCommitWait);
}

enum class PreferredCopyInstruction {
  kAuto,
  kTMA,
  kCPAsync,
  kAIU,
  kSync,
};

constexpr const char *kPreferInstruction = "prefer_instruction";

constexpr std::pair<const char *, PreferredCopyInstruction>
    kPreferredCopyInstructions[] = {
        {"tma", PreferredCopyInstruction::kTMA},
        {"cp_async", PreferredCopyInstruction::kCPAsync},
        {"aiu", PreferredCopyInstruction::kAIU},
        {"sync", PreferredCopyInstruction::kSync},
};

std::optional<std::string> GetStringAnnotation(const CopyNode &op,
                                               const char *key) {
  auto val = op.annotations.Get(key);
  if (!val) {
    return std::nullopt;
  }
  auto str = val->as<StringImmNode>();
  ICHECK(str) << "T.copy " << key << " annotation must be a string, but got "
              << val.value().GetTypeKey();
  return str->value;
}

PreferredCopyInstruction
ParsePreferredCopyInstruction(const std::string &prefer) {
  for (const auto &[name, inst] : kPreferredCopyInstructions) {
    if (prefer == name) {
      return inst;
    }
  }
  LOG(FATAL)
      << "Unsupported T.copy prefer_instruction=\"" << prefer
      << "\". Expected one of: \"tma\", \"cp_async\", \"aiu\", \"sync\".";
  return PreferredCopyInstruction::kAuto;
}

PreferredCopyInstruction GetPreferredInstruction(const CopyNode &op) {
  if (auto prefer = GetStringAnnotation(op, kPreferInstruction)) {
    return ParsePreferredCopyInstruction(prefer.value());
  }
  return PreferredCopyInstruction::kAuto;
}

bool CheckLDSMCopy(const CopyNode &op, Target target) {
  return TargetPPUHasLdmatrix(target) && IsSharedBuffer(op.src) &&
         IsFragmentBuffer(op.dst);
}

bool CheckAiuLoad(const CopyNode &op, Target target) {
  return TargetIsPPU(target) && TargetHasAiuCopy(target) &&
         IsGlobalBuffer(op.src) && IsSharedBuffer(op.dst) &&
         op.src->dtype == op.dst->dtype &&
         (op.src->dtype.is_float16() || op.src->dtype.is_bfloat16() ||
          op.src->dtype.is_float8_e4m3fn() || op.src->dtype.is_float8_e5m2() ||
          op.src->dtype.is_float4_e2m1fn());
}

bool CheckCPAsyncCopyPreconditions(const CopyNode &op) {
  return IsGlobalBuffer(op.src) && IsSharedBuffer(op.dst) &&
         op.src->dtype == op.dst->dtype;
}

bool CheckCPAsyncCopy(const CopyNode &op, Target target,
                      const LayoutMap &layout_map, arith::Analyzer *analyzer) {
  if (!TargetPPUHasAsyncCopy(target)) {
    return false;
  }
  if (!CheckCPAsyncCopyPreconditions(op)) {
    return false;
  }
  // Skip vectorize size checks here because the layout is not stable during
  // layout inference and transform classification.
  return true;
}

} // namespace

const char *CopyInstToString(CopyInst inst) {
  switch (inst) {
  case CopyInst::kNormal:
    return "Normal";
  case CopyInst::kLDSM:
    return "LDSM";
  case CopyInst::kCPAsync:
    return "CPAsync";
  case CopyInst::kAiuLoad:
    return "AiuLoad";
  case CopyInst::kInvalid:
    return "Invalid";
  default:
    return "Unknown";
  }
}

bool CopyInstIsCPAsync(CopyInst inst) { return inst == CopyInst::kCPAsync; }

std::optional<AIUWarpPartition>
FindAIUWarpPartition(bool swizzled, int swizzle_bytes, int64_t height,
                     int64_t width_bytes, int num_warps) {
  if (num_warps < 1) {
    return std::nullopt;
  }
  int64_t num_blocks = swizzled ? width_bytes / swizzle_bytes : 1;
  if (num_warps == 1) {
    if (swizzled && height % 8 != 0) {
      return std::nullopt;
    }
    return AIUWarpPartition{1, 1, static_cast<int>(num_blocks),
                            static_cast<int>(height)};
  }
  const int max_col_splits =
      static_cast<int>(std::min<int64_t>(num_warps, num_blocks));
  for (int col_splits = max_col_splits; col_splits >= 1; --col_splits) {
    if (swizzled && num_blocks % col_splits != 0) {
      continue;
    }
    if (!swizzled && col_splits != 1) {
      continue;
    }
    const int max_row_splits =
        std::min(num_warps / col_splits,
                 static_cast<int>(std::min<int64_t>(height, INT_MAX)));
    for (int row_splits = max_row_splits; row_splits >= 1; --row_splits) {
      if (height % row_splits != 0) {
        continue;
      }
      int rows_per_warp = static_cast<int>(height / row_splits);
      if (swizzled && rows_per_warp % 8 != 0) {
        continue;
      }
      int blocks_per_warp =
          swizzled ? static_cast<int>(num_blocks / col_splits) : 1;
      return AIUWarpPartition{row_splits, col_splits, blocks_per_warp,
                              rows_per_warp};
    }
  }
  return std::nullopt;
}

namespace {

struct CopyFacts {
  bool target_supported = false;
  bool explicit_tma = false;
  bool explicit_cp_async = false;
  bool no_implicit_async_commit_wait = false;
  PreferredCopyInstruction prefer_instruction = PreferredCopyInstruction::kAuto;
  bool pass_context_disables_aiu = true;
  int64_t cluster_mask = 0;
  bool can_cp_async = false;
  bool can_aiu_load = false;
  bool can_ldsm = false;
  std::string async_unavailable_reason;
};

bool IsPpuCopyTarget(Target target) {
  return target.defined() && TargetIsPPU(target);
}

CopyInstSelection Supported(CopyInst inst) {
  return CopyInstSelection{inst, true, ""};
}

CopyInstSelection Unsupported(std::string reason) {
  return CopyInstSelection{CopyInst::kInvalid, false, std::move(reason)};
}

std::string MakeTmaUnavailableReason(const CopyNode &op) {
  std::ostringstream oss;
  oss << "PPU only supports ppu0010/ppu0015; TMA bulk copy, T.tma_copy(), "
         "gather4/scatter4, and cluster-copy paths require ppu0015+ TMA. Got "
         "src="
      << op.src->name << " (scope=" << op.src.scope()
      << ", dtype=" << op.src->dtype << "), dst=" << op.dst->name
      << " (scope=" << op.dst.scope() << ", dtype=" << op.dst->dtype << ").";
  return oss.str();
}

std::string MakeAsyncUnavailableReason(const CopyNode &op, Target target) {
  std::ostringstream oss;
  if (!target.defined()) {
    oss << "T.async_copy requires a defined target.";
  } else if (!TargetPPUHasAsyncCopy(target)) {
    oss << "T.async_copy is only supported on targets with cp.async support "
           "(PPU0010+). Got target="
        << target;
  } else if (!IsGlobalBuffer(op.src) || !IsSharedBuffer(op.dst)) {
    oss << "T.async_copy only supports global->shared/shared.dyn copies. "
           "Got src="
        << op.src->name << " (scope=" << op.src.scope()
        << "), dst=" << op.dst->name << " (scope=" << op.dst.scope() << ").";
  } else if (op.src->dtype != op.dst->dtype) {
    oss << "T.async_copy requires equal byte-addressable dtypes. Got src "
           "dtype="
        << op.src->dtype << ", dst dtype=" << op.dst->dtype << ".";
  } else {
    oss << "Explicit async copy semantics require cp.async lowering, but "
           "constraints were not satisfied. Got src="
        << op.src->name << " (scope=" << op.src.scope()
        << ", dtype=" << op.src->dtype << "), dst=" << op.dst->name
        << " (scope=" << op.dst.scope() << ", dtype=" << op.dst->dtype << ").";
  }
  return oss.str();
}

bool IsAutoAsyncCopyEnabled(bool default_enabled) {
  using namespace tvm::transform;
  PassContext pass_ctx = PassContext::Current();
  return pass_ctx->GetConfig<Bool>(kEnableAsyncCopy, Bool(default_enabled))
      .value();
}

CopyInst SelectSyncLikeInst(const CopyFacts &facts) {
  if (facts.can_ldsm) {
    return CopyInst::kLDSM;
  }
  return CopyInst::kNormal;
}

CopyFacts AnalyzeCopyFacts(const CopyNode &op, const CopyAnalysisContext &ctx) {
  CopyFacts facts;
  facts.target_supported = IsPpuCopyTarget(ctx.target);
  facts.explicit_tma = GetIsTmaCopy(op);
  facts.explicit_cp_async = GetIsAsyncCopy(op);
  facts.no_implicit_async_commit_wait = GetNoImplicitAsyncCommitWait(op);
  facts.prefer_instruction = GetPreferredInstruction(op);
  facts.pass_context_disables_aiu =
      tvm::transform::PassContext::Current()
          ->GetConfig<Bool>(kDisableAIULower, Bool(true))
          .value();
  facts.cluster_mask = GetClusterMask(op);
  facts.async_unavailable_reason = MakeAsyncUnavailableReason(op, ctx.target);

  if (!facts.target_supported) {
    return facts;
  }

  arith::Analyzer local_analyzer;
  arith::Analyzer *analyzer =
      ctx.analyzer != nullptr ? ctx.analyzer : &local_analyzer;
  static const LayoutMap empty_layout_map;
  const LayoutMap &layout_map =
      ctx.layout_map != nullptr ? *ctx.layout_map : empty_layout_map;

  facts.can_cp_async = CheckCPAsyncCopy(op, ctx.target, layout_map, analyzer);
  facts.can_aiu_load = CheckAiuLoad(op, ctx.target);
  facts.can_ldsm = CheckLDSMCopy(op, ctx.target);
  return facts;
}

} // namespace

CopyInstSelection SelectCopyInstForLowering(const CopyNode &op,
                                            const CopyAnalysisContext &ctx) {
  CopyFacts facts = AnalyzeCopyFacts(op, ctx);
  std::string tma_reason = MakeTmaUnavailableReason(op);

  if (GetBoolAnnotation(op, "is_gather4") ||
      GetBoolAnnotation(op, "is_scatter4")) {
    return Unsupported(tma_reason);
  }

  if (facts.cluster_mask != 0) {
    return Unsupported(tma_reason);
  }

  if (facts.explicit_tma) {
    return Unsupported(tma_reason);
  }

  if (facts.explicit_cp_async) {
    return facts.can_cp_async ? Supported(CopyInst::kCPAsync)
                              : Unsupported(facts.async_unavailable_reason);
  }

  if (facts.no_implicit_async_commit_wait) {
    if (facts.prefer_instruction == PreferredCopyInstruction::kAIU &&
        !facts.pass_context_disables_aiu && facts.can_aiu_load) {
      return Supported(CopyInst::kAiuLoad);
    }
    return facts.can_cp_async ? Supported(CopyInst::kCPAsync)
                              : Unsupported(facts.async_unavailable_reason);
  }

  if (facts.prefer_instruction == PreferredCopyInstruction::kTMA) {
    return Unsupported("T.copy prefer_instruction=\"tma\" is unsupported on "
                       "PPU ppu0010/ppu0015. " +
                       tma_reason);
  }

  if (facts.prefer_instruction == PreferredCopyInstruction::kCPAsync) {
    if (!IsAutoAsyncCopyEnabled(/*default_enabled=*/true)) {
      return Unsupported(
          "T.copy prefer_instruction=\"cp_async\" conflicts with "
          "pass config tl.enable_async_copy=false.");
    }
    return facts.can_cp_async
               ? Supported(CopyInst::kCPAsync)
               : Unsupported("T.copy prefer_instruction="
                             "\"cp_async\" could not be honored: " +
                             facts.async_unavailable_reason);
  }

  if (facts.prefer_instruction == PreferredCopyInstruction::kAIU) {
    if (facts.pass_context_disables_aiu) {
      return Supported(SelectSyncLikeInst(facts));
    }
    if (facts.can_aiu_load)
      return Supported(CopyInst::kAiuLoad);
    if (facts.can_cp_async)
      return Supported(CopyInst::kCPAsync);
    return Supported(SelectSyncLikeInst(facts));
  }

  if (facts.prefer_instruction == PreferredCopyInstruction::kSync) {
    return Supported(SelectSyncLikeInst(facts));
  }

  return Supported(SelectSyncLikeInst(facts));
}

} // namespace ppu
} // namespace tl
} // namespace tvm
