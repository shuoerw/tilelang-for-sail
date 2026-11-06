/*!
 * \file schedule_pipeline_copies.cc
 * \brief Post-place independent PPU pipeline copy groups after GEMM.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>
#include <unordered_set>

#include "op/builtin.h"
#include "op/copy.h"
#include "op/gemm.h"
#include "op/utils.h"
#include "ppu/target_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

using BufferMap =
    std::unordered_map<Var, Buffer, ObjectPtrHash, ObjectPtrEqual>;
using VarSet = std::unordered_set<Var, ObjectPtrHash, ObjectPtrEqual>;

// Block regions may conservatively cover every pipeline slot. Inspect the
// underlying tile op instead, while retaining its original block metadata.
const CallNode *TileCall(Stmt stmt, VarSet *bindings = nullptr) {
  while (true) {
    if (const auto *realize = stmt.as<SBlockRealizeNode>()) {
      const auto &block = realize->block;
      if (!is_one(realize->predicate) || !realize->iter_values.empty() ||
          !block->iter_vars.empty() || block->init.defined() ||
          !block->alloc_buffers.empty() || !block->match_buffers.empty() ||
          !block->annotations.empty())
        return nullptr;
      stmt = block->body;
    } else if (const auto *seq = stmt.as<SeqStmtNode>()) {
      if (!bindings || seq->seq.empty())
        return nullptr;
      // Replayed scalar indices travel with their producer. They must not
      // read memory or define values needed by the preceding consumer.
      for (size_t i = 0; i + 1 < seq->seq.size(); ++i) {
        const auto *bind = seq->seq[i].as<BindNode>();
        if (!bind || SideEffect(bind->value) != CallEffectKind::kPure)
          return nullptr;
        bindings->insert(bind->var);
      }
      stmt = seq->seq.back();
    } else {
      break;
    }
  }
  const auto *evaluate = stmt.as<EvaluateNode>();
  return evaluate ? evaluate->value.as<CallNode>() : nullptr;
}

bool PureRanges(const Array<Range> &ranges) {
  for (const Range &range : ranges) {
    if (SideEffect(range->min) != CallEffectKind::kPure ||
        SideEffect(range->extent) != CallEffectKind::kPure)
      return false;
  }
  return true;
}

bool PureAddress(const BufferRegion &region) {
  const Buffer &buffer = region->buffer;
  if (!PureRanges(region->region) ||
      SideEffect(buffer->elem_offset) != CallEffectKind::kPure)
    return false;
  for (const PrimExpr &stride : buffer->strides) {
    if (SideEffect(stride) != CallEffectKind::kPure)
      return false;
  }
  for (const PrimExpr &size : buffer->shape) {
    if (SideEffect(size) != CallEffectKind::kPure)
      return false;
  }
  return true;
}

bool PureScalarArguments(const CallNode *call) {
  for (const PrimExpr &arg : call->args) {
    const auto *region_call = arg.as<CallNode>();
    if (region_call && region_call->op.same_as(region()))
      continue;
    if (SideEffect(arg) != CallEffectKind::kPure)
      return false;
  }
  return true;
}

bool TrueAnnotation(const CopyNode &copy, const char *key) {
  auto value = copy.annotations.Get(key);
  const auto *imm = value ? value.value().as<IntImmNode>() : nullptr;
  return imm && imm->value != 0;
}

class PipelineCopyScheduler : public StmtMutator {
public:
  explicit PipelineCopyScheduler(const Stmt &body) {
    auto record = [&](const Buffer &buffer) {
      if (ambiguous_.count(buffer->data))
        return;
      if (owned_.count(buffer->data)) {
        owned_.erase(buffer->data);
        ambiguous_.insert(buffer->data);
      } else {
        owned_.emplace(buffer->data, buffer);
      }
    };
    PostOrderVisit(body, [&](const ObjectRef &node) {
      if (const auto *block = node.as<SBlockNode>()) {
        for (const Buffer &buffer : block->alloc_buffers)
          record(buffer);
        has_unsupported_annotations_ |=
            block->annotations.count(attr::kLayoutMap) != 0 ||
            block->annotations.count(attr::kSafeValueMap) != 0;
      } else if (const auto *alloc = node.as<AllocBufferNode>()) {
        record(alloc->buffer);
      } else if (const auto *attribute = node.as<AttrStmtNode>()) {
        has_unsupported_annotations_ |=
            attribute->attr_key == attr::kLayoutMap ||
            attribute->attr_key == attr::kSafeValueMap ||
            attribute->attr_key == "volatile_scope";
      }
    });
  }

private:
  bool Owned(const Buffer &buffer) const {
    auto it = owned_.find(buffer->data);
    return it != owned_.end() && it->second.same_as(buffer);
  }

  bool DenseRing(const BufferRegion &region, int64_t stages) {
    const Buffer &buffer = region->buffer;
    if (!Owned(buffer) || !IsSharedBuffer(buffer) || buffer->shape.size() < 2 ||
        region->region.size() != buffer->shape.size() ||
        !buffer->strides.empty() || !is_zero(buffer->elem_offset) ||
        !buffer->dtype.is_scalar() ||
        !analyzer_.CanProveEqual(buffer->shape[0],
                                 IntImm(buffer->shape[0].dtype(), stages)) ||
        !analyzer_.CanProveEqual(region->region[0]->extent, 1))
      return false;
    for (size_t axis = 0; axis < region->region.size(); ++axis) {
      const Range &range = region->region[axis];
      if (!analyzer_.CanProve(range->min >= 0) ||
          !analyzer_.CanProve(range->extent >= 0) ||
          !analyzer_.CanProve(range->min + range->extent <=
                              buffer->shape[axis]))
        return false;
    }
    // Distinct FP4 slots must also occupy distinct physical bytes.
    int64_t bits_mod_byte = buffer->dtype.bits() % 8;
    for (size_t axis = 1; axis < buffer->shape.size(); ++axis) {
      const auto *size = buffer->shape[axis].as<IntImmNode>();
      if (!size || size->value <= 0)
        return false;
      bits_mod_byte = bits_mod_byte * (size->value % 8) % 8;
    }
    return bits_mod_byte == 0;
  }

  bool SeparateSlots(const BufferRegion &write, const BufferRegion &read,
                     int64_t stages) {
    if (!write->buffer.same_as(read->buffer) || !DenseRing(read, stages))
      return false;
    const auto *write_slot = write->region[0]->min.as<FloorModNode>();
    const auto *read_slot = read->region[0]->min.as<FloorModNode>();
    if (!write_slot || !read_slot ||
        !analyzer_.CanProveEqual(write_slot->b,
                                 IntImm(write_slot->b.dtype(), stages)) ||
        !analyzer_.CanProveEqual(read_slot->b,
                                 IntImm(read_slot->b.dtype(), stages)))
      return false;
    PrimExpr distance = analyzer_.Simplify(
        floormod(write_slot->a - read_slot->a, write_slot->b));
    const auto *imm = distance.as<IntImmNode>();
    return imm && imm->value != 0;
  }

  Stmt VisitStmt_(const ForNode *op) final {
    For loop = Downcast<For>(StmtMutator::VisitStmt_(op));
    auto depth = loop->annotations.Get("tl_pipelined_num_stages");
    const auto *stages = depth ? depth.value().as<IntImmNode>() : nullptr;
    const auto *seq = loop->body.as<SeqStmtNode>();
    if (has_unsupported_annotations_ || !stages || stages->value < 2 || !seq ||
        seq->seq.size() < 4 || loop->kind != ForKind::kSerial ||
        (loop->step.defined() && !is_one(loop->step.value())))
      return loop;

    // Match one complete compiler-managed group followed by its consumer.
    const size_t copies = seq->seq.size() - 3;
    const auto *commit_stmt = seq->seq[copies].as<EvaluateNode>();
    const auto *wait_stmt = seq->seq[copies + 1].as<EvaluateNode>();
    const auto *commit =
        commit_stmt ? commit_stmt->value.as<CallNode>() : nullptr;
    const auto *wait = wait_stmt ? wait_stmt->value.as<CallNode>() : nullptr;
    const auto *gemm_call = TileCall(seq->seq.back());
    if (!commit || !wait || !gemm_call ||
        !commit->op.same_as(builtin::ptx_commit_group()) ||
        !commit->args.empty() || !wait->op.same_as(builtin::ptx_wait_group()) ||
        wait->args.size() != 1)
      return loop;
    const auto *count = wait->args[0].as<IntImmNode>();
    if (!count || count->value != stages->value - 1)
      return loop;
    TileOperator consumer = ParseOperator(GetRef<Call>(gemm_call));
    if (!consumer.defined() || !consumer.as<GemmNode>() ||
        !PureScalarArguments(gemm_call))
      return loop;
    AccessRegions access = consumer->GetAccessRegions();
    for (const BufferRegion &write : access.writes) {
      if (!Owned(write->buffer) || !PureAddress(write) ||
          !(IsLocalBuffer(write->buffer) || IsFragmentBuffer(write->buffer)))
        return loop;
    }
    for (const BufferRegion &read : access.reads) {
      if (!PureAddress(read) ||
          (IsSharedBuffer(read->buffer) && !DenseRing(read, stages->value)))
        return loop;
    }
    VarSet bindings;
    for (size_t i = 0; i < copies; ++i) {
      const auto *call = TileCall(seq->seq[i], &bindings);
      if (!call)
        return loop;
      TileOperator producer = ParseOperator(GetRef<Call>(call));
      const auto *copy = producer.as<CopyNode>();
      if (!copy || !IsGlobalBuffer(copy->src) ||
          copy->src->dtype != copy->dst->dtype ||
          !TrueAnnotation(*copy, attr::kAsyncCopyNoImplicitCommitWait) ||
          TrueAnnotation(*copy, "is_async_copy") ||
          TrueAnnotation(*copy, "is_tma_copy") || copy->dst_block.defined() ||
          TrueAnnotation(*copy, "force_cp_async") ||
          !PureScalarArguments(call) ||
          !PureAddress(BufferRegion(copy->src, copy->src_range)))
        return loop;
      BufferRegion destination(copy->dst, copy->dst_range);
      if (!DenseRing(destination, stages->value) || !PureAddress(destination))
        return loop;
      bool consumed = false;
      for (const BufferRegion &read : access.reads) {
        if (!read->buffer->data.same_as(copy->dst->data))
          continue;
        if (!SeparateSlots(destination, read, stages->value))
          return loop;
        consumed = true;
      }
      if (!consumed)
        return loop;
    }
    if (UsesVar(GetRef<Call>(gemm_call), [&](const VarNode *var) {
          return bindings.count(GetRef<Var>(var)) != 0;
        }))
      return loop;

    // Moving this iteration's commit after the wait removes one newer group
    // from the FIFO. Waiting for one fewer group preserves the completed
    // prefix.
    Call new_wait = GetRef<Call>(wait);
    new_wait.CopyOnWrite()->args.Set(0, IntImm(count->dtype, count->value - 1));
    Evaluate new_wait_stmt = GetRef<Evaluate>(wait_stmt);
    new_wait_stmt.CopyOnWrite()->value = std::move(new_wait);
    Array<Stmt> ordered{new_wait_stmt, seq->seq.back()};
    for (size_t i = 0; i <= copies; ++i)
      ordered.push_back(seq->seq[i]);
    loop.CopyOnWrite()->body = SeqStmt(ordered, seq->span);
    return loop;
  }

  BufferMap owned_;
  VarSet ambiguous_;
  arith::Analyzer analyzer_;
  bool has_unsupported_annotations_{false};
};

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  auto pass_func = [](PrimFunc func, const IRModule &,
                      const tvm::transform::PassContext &) {
    auto target = func->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsPPU(target.value()) ||
        (func->attrs.defined() &&
         (func->attrs->dict.count(attr::kLayoutMap) ||
          func->attrs->dict.count(attr::kSafeValueMap))))
      return func;
    Stmt body = PipelineCopyScheduler(func->body)(func->body);
    if (!body.same_as(func->body))
      func.CopyOnWrite()->body = std::move(body);
    return func;
  };
  reflection::GlobalDef().def(
      "tl.ppu.transform.SchedulePipelineCopies", [pass_func]() {
        return tirx::transform::CreatePrimFuncPass(
            pass_func, 0, "tl.ppu.SchedulePipelineCopies", {});
      });
}

} // namespace tl
} // namespace tvm
