/*!
 * \file register_pingpong_pipeline.cc
 * \brief Register ping-pong software pipeline for the inner ki loop.
 *
 * The transformation:
 *   1. Find For loops annotated with "register_pipeline_stage"
 *   2. Collect ping-pong candidate buffers (ldmatrix dests + shared→local
 * loads)
 *   3. Split loop body into load_group / compute_group
 *   4. Fully unroll into prologue → steady-state → epilogue
 *   5. Route each iteration's buffer accesses to per-slot independent buffers
 *   6. Replace original allocations with per-slot allocations in the SBlock
 *
 * Each SBlock containing an annotated For loop is processed independently,
 * so both main-loop and epilogue stages are correctly ping-pong unrolled.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/memory.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "op/builtin.h"
#include "tirx/transform/ir_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;
using ffi::Array;
using ffi::GetRef;
using ffi::Map;
using ffi::ObjectPtr;
using ffi::ObjectRef;
using ffi::Optional;
using ffi::String;

namespace {

using VarSet = std::unordered_set<Var, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;
using BufferSlotMap =
    std::unordered_map<Var, Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>;

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

static constexpr const char *kRegisterPipelineStage = "register_pipeline_stage";

// ldmatrix intrinsics whose destination local buffer is a ping-pong candidate.
static bool IsLdmatrixOp(const std::string &name) {
  return name == "tl.tix_ldmatrix_swzl" || name == "tirx.ptx_ldmatrix" ||
         name == "tl.ptx_ldmatrix" || name == "tl.tix_ldmatrix_b8" ||
         name == "tir.ptx_ldmatrix";
}

// MMA ops that reference ping-pong buffers via buf.data arguments.
static bool IsMmaOp(const std::string &name) {
  return name == "tirx.ptx_mma" || name == "tir.ptx_mma" ||
         name == "tl.ptx_mma" || name == "tirx.ptx_mma_blockscaled" ||
         name == "tir.ptx_mma_blockscaled" || name == "tl.ptx_mma_blockscaled";
}

static std::string GetOpName(const RelaxExpr &op_expr) {
  if (const auto *op = op_expr.as<OpNode>()) {
    return op->name;
  }
  return "";
}

// Extract an int64 from an annotation value. The value may be stored either as
// a raw POD int64 (when passed from Python as a plain int through the FFI
// boundary) or as a wrapped IntImmNode object. Returns nullopt if neither.
static ffi::Optional<int64_t> GetAnnotationInt(const ffi::Any &val) {
  // Case 1: raw POD int64 (Python int through FFI)
  if (auto as_i64 = val.as<int64_t>()) {
    return as_i64.value();
  }
  // Case 2: wrapped IntImmNode object
  if (const auto *imm = val.as<IntImmNode>()) {
    return imm->value;
  }
  return ffi::Optional<int64_t>();
}

// ---------------------------------------------------------------------------
// Buffer scope helper
// ---------------------------------------------------------------------------

static std::string GetBufferScope(const Buffer &buf) {
  if (auto ptr_type = buf->data->type_annotation.as<PointerTypeNode>()) {
    return ptr_type->storage_scope;
  }
  return "";
}

// ---------------------------------------------------------------------------
// 1. LdmatrixDestCollector
// ---------------------------------------------------------------------------

class LdmatrixDestCollector : public StmtExprVisitor {
public:
  VarSet dest_data_vars;

  void VisitExpr_(const CallNode *op) final {
    std::string name = GetOpName(op->op);
    if (IsLdmatrixOp(name) && op->args.size() > 0) {
      // The regular ldmatrix call ends with its destination access pointer,
      // while the six-argument swizzled form carries mode and trans_block
      // after the destination.
      const size_t dst_index =
          name == "tl.tix_ldmatrix_swzl" && op->args.size() > 3
              ? 3
              : op->args.size() - 1;
      RecordFromAccessPtr(op->args[dst_index]);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

private:
  void RecordFromAccessPtr(const PrimExpr &expr) {
    PostOrderVisit(expr, [this](const ObjectRef &obj) {
      if (const auto *load = obj.as<BufferLoadNode>()) {
        dest_data_vars.insert(load->buffer->data);
      }
    });
  }
};

// ---------------------------------------------------------------------------
// 2. SharedToLocalLoadCollector
// ---------------------------------------------------------------------------

class SharedToLocalLoadCollector : public StmtExprVisitor {
public:
  VarSet dest_data_vars;

  void VisitStmt_(const BufferStoreNode *op) final {
    std::string scope = GetBufferScope(op->buffer);
    if (scope == "local" && ExprReadsNonLocal(op->value)) {
      dest_data_vars.insert(op->buffer->data);
    }
    StmtExprVisitor::VisitStmt_(op);
  }

private:
  static bool ExprReadsNonLocal(const PrimExpr &expr) {
    bool found = false;
    PostOrderVisit(expr, [&found](const ObjectRef &obj) {
      if (found)
        return;
      if (const auto *load = obj.as<BufferLoadNode>()) {
        std::string scope = GetBufferScope(load->buffer);
        if (scope != "local") {
          found = true;
        }
      }
    });
    return found;
  }
};

// ---------------------------------------------------------------------------
// 3. BufferRouteInjector — routes buffer accesses to per-slot buffers
// ---------------------------------------------------------------------------

class BufferRouteInjector : public StmtExprMutator {
public:
  // slot_map: original data var → slot Buffer
  explicit BufferRouteInjector(const BufferSlotMap &slot_map)
      : slot_map_(slot_map) {}

  Buffer VisitBufferUse(const Buffer &buffer) final {
    Buffer routed = StmtExprMutator::VisitBufferUse(buffer);
    auto it = slot_map_.find(routed->data);
    if (it != slot_map_.end()) {
      // Preserve view dtype and indexing; only change the storage slot.
      routed.CopyOnWrite()->data = it->second->data;
      buffer_remap_.Set(buffer, routed);
    }
    return routed;
  }

  PrimExpr VisitExpr_(const VarNode *op) final {
    auto it = slot_map_.find(GetRef<Var>(op));
    return it == slot_map_.end() ? GetRef<Var>(op) : it->second->data;
  }

private:
  const BufferSlotMap &slot_map_;
};

// ---------------------------------------------------------------------------
// Helper: check if a statement contains ldmatrix ops
// ---------------------------------------------------------------------------

static bool StmtContainsLdmatrixOp(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&found](const ObjectRef &obj) {
    if (found)
      return;
    if (const auto *call = obj.as<CallNode>()) {
      if (IsLdmatrixOp(GetOpName(call->op))) {
        found = true;
      }
    }
  });
  return found;
}

// Helper: check if a statement writes to any of the given data vars
static bool StmtWritesAny(const Stmt &stmt, const VarSet &data_vars) {
  bool found = false;
  PostOrderVisit(stmt, [&found, &data_vars](const ObjectRef &obj) {
    if (found)
      return;
    if (const auto *store = obj.as<BufferStoreNode>()) {
      if (data_vars.count(store->buffer->data)) {
        found = true;
      }
    }
  });
  return found;
}

// A statement is a "load" if it issues ldmatrix OR stores into a ping-pong
// buffer
static bool StmtIsLoad(const Stmt &stmt, const VarSet &pingpong_vars) {
  return StmtContainsLdmatrixOp(stmt) || StmtWritesAny(stmt, pingpong_vars);
}

// Slot storage must be private to the rewritten loop, including through views
// and raw pointers used by surrounding initialization or epilogue statements.
static bool ReferencesAny(const Stmt &stmt, const VarSet &data_vars) {
  class ReferenceVisitor : public StmtExprVisitor {
  public:
    explicit ReferenceVisitor(const VarSet &data_vars)
        : data_vars_(data_vars) {}
    bool found = false;

    void VisitExpr_(const VarNode *op) final {
      found |= data_vars_.count(GetRef<Var>(op)) != 0;
    }

    void VisitBufferUse(const Buffer &buffer) final {
      if (!visited_.insert(buffer).second)
        return;
      found |= data_vars_.count(buffer->data) != 0;
      StmtExprVisitor::VisitBufferDef(buffer, false);
    }

    void VisitBufferDef(const Buffer &buffer, bool) final {
      VisitBufferUse(buffer);
    }

  private:
    const VarSet &data_vars_;
    std::unordered_set<Buffer, ffi::ObjectPtrHash, ffi::ObjectPtrEqual>
        visited_;
  } visitor(data_vars);
  visitor(stmt);
  return visitor.found;
}

// Only reorder understood memory operations. Unknown calls, barriers, and
// conditional definitions cannot establish an independent next-iteration load.
class PipelineAccesses : public StmtExprVisitor {
public:
  bool safe = true;
  VarSet reads, writes;

  void VisitExpr_(const VarNode *op) final {
    if (op->dtype.is_handle())
      reads.insert(GetRef<Var>(op));
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    reads.insert(op->buffer->data);
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    writes.insert(op->buffer->data);
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt(const Stmt &stmt) final {
    if (!stmt.as<SeqStmtNode>() && !stmt.as<ForNode>() &&
        !stmt.as<BufferStoreNode>() && !stmt.as<EvaluateNode>()) {
      safe = false;
      return;
    }
    StmtExprVisitor::VisitStmt(stmt);
  }

  void VisitExpr_(const CallNode *op) final {
    const std::string name = GetOpName(op->op);
    if (IsLdmatrixOp(name)) {
      const size_t dst =
          name == "tl.tix_ldmatrix_swzl" ? 3 : op->args.size() - 1;
      if (dst >= op->args.size()) {
        safe = false;
        return;
      }
      bool found = false;
      PostOrderVisit(op->args[dst], [&](const ObjectRef &obj) {
        if (const auto *load = obj.as<BufferLoadNode>()) {
          found = true;
          writes.insert(load->buffer->data);
          for (const PrimExpr &index : load->indices)
            VisitExpr(index);
        }
      });
      safe &= found;
      for (size_t i = 0; i < op->args.size(); ++i) {
        if (i != dst)
          VisitExpr(op->args[i]);
      }
      return;
    }
    if (IsMmaOp(name)) {
      // Treat every pointer operand conservatively as both read and written.
      for (const PrimExpr &arg : op->args) {
        if (!arg.dtype().is_handle() || arg.as<StringImmNode>())
          continue;
        const auto *var = arg.as<VarNode>();
        if (!var) {
          safe = false;
          return;
        }
        reads.insert(GetRef<Var>(var));
        writes.insert(GetRef<Var>(var));
      }
    } else if (name != "tl.access_ptr") {
      // Inspect the callee's effect separately from argument reads. Pure
      // register operations such as byte_perm may take BufferLoad arguments;
      // the normal visitor records those reads for dependency checks.
      static auto effects = Op::GetAttrMap<TCallEffectKind>("TCallEffectKind");
      auto callee = op->op.as<Op>();
      if (!callee || !effects.count(callee.value()) ||
          effects[callee.value()]->value !=
              static_cast<int>(CallEffectKind::kPure))
        safe = false;
    }
    StmtExprVisitor::VisitExpr_(op);
  }
};

// Only fixed register footprints can be split into independent iterations.
// Track physical bits so typed views retain the same storage interpretation.
class RegisterFootprints : public StmtExprVisitor {
public:
  using Span = std::pair<int64_t, int64_t>;
  using Regions = std::unordered_map<Var, std::vector<Span>, ffi::ObjectPtrHash,
                                     ffi::ObjectPtrEqual>;
  bool safe = true;
  Regions reads, writes;

  explicit RegisterFootprints(const BufferSlotMap &allocations)
      : allocations_(allocations) {}

  bool ReadsCoveredBy(const RegisterFootprints &loads) const {
    if (!safe || !loads.safe)
      return false;
    for (const auto &[var, required] : reads) {
      auto it = loads.writes.find(var);
      if (it == loads.writes.end())
        return false;
      auto defined = it->second;
      std::sort(defined.begin(), defined.end());
      for (const auto &[begin, end] : required) {
        int64_t covered = begin;
        for (const auto &[lo, hi] : defined) {
          if (lo > covered)
            break;
          covered = std::max(covered, hi);
        }
        if (covered < end)
          return false;
      }
    }
    return true;
  }

private:
  void VisitBufferUse(const Buffer &buffer) final {
    auto check = [&](const PrimExpr &expr) {
      if (SideEffect(expr) != CallEffectKind::kPure ||
          UsesVar(expr, [&](const VarNode *var) {
            return allocations_.count(GetRef<Var>(var)) != 0;
          }))
        safe = false;
    };
    check(buffer->elem_offset);
    for (const PrimExpr &dim : buffer->shape)
      check(dim);
    for (const PrimExpr &stride : buffer->strides)
      check(stride);
  }

  void Add(Regions *regions, const Buffer &buffer,
           const Array<PrimExpr> &indices, PrimExpr count) {
    auto allocation = allocations_.find(buffer->data);
    if (allocation == allocations_.end())
      return;
    const Buffer &storage = allocation->second;
    if (!storage->strides.empty() || !is_zero(storage->elem_offset) ||
        !storage->dtype.is_scalar() || !buffer->dtype.is_scalar() ||
        indices.size() != buffer->shape.size()) {
      safe = false;
      return;
    }
    PrimExpr size = Integer(storage->dtype.bits());
    for (const PrimExpr &dim : storage->shape)
      size = size * dim;
    PrimExpr offset = buffer->elem_offset;
    PrimExpr stride = Integer(1);
    for (int axis = static_cast<int>(indices.size()) - 1; axis >= 0; --axis) {
      offset = offset +
               indices[axis] *
                   (buffer->strides.empty() ? stride : buffer->strides[axis]);
      stride = stride * buffer->shape[axis];
    }
    auto total = analyzer_.Simplify(size).as<IntImm>();
    auto start = analyzer_.Simplify(offset * buffer->dtype.bits()).as<IntImm>();
    auto bits = analyzer_.Simplify(count * buffer->dtype.bits()).as<IntImm>();
    if (!total || !start || !bits || start.value()->value < 0 ||
        bits.value()->value <= 0 ||
        start.value()->value > total.value()->value - bits.value()->value) {
      safe = false;
      return;
    }
    (*regions)[buffer->data].emplace_back(
        start.value()->value, start.value()->value + bits.value()->value);
  }

  void Pointer(const PrimExpr &expr, Regions *regions,
               int64_t instruction_bits = 0) {
    const auto *ptr = expr.as<CallNode>();
    const auto *load =
        ptr && GetOpName(ptr->op) == "tl.access_ptr" && ptr->args.size() == 3
            ? ptr->args[0].as<BufferLoadNode>()
            : nullptr;
    if (!load) {
      safe = false;
      return;
    }
    if (instruction_bits &&
        !analyzer_.CanProveEqual(ptr->args[1] * load->buffer->dtype.bits(),
                                 Integer(instruction_bits))) {
      safe = false;
      return;
    }
    VisitBufferUse(load->buffer);
    Add(regions, load->buffer, load->indices, ptr->args[1]);
  }

  void VisitStmt_(const ForNode *op) final {
    const auto *minimum = op->min.as<IntImmNode>();
    const auto *extent = op->extent.as<IntImmNode>();
    if (!minimum || !extent || extent->value < 0 ||
        extent->value > remaining_iterations_ ||
        (op->step.defined() && !is_one(op->step.value())) ||
        (op->kind != ForKind::kSerial && op->kind != ForKind::kUnrolled)) {
      safe = false;
      return;
    }
    remaining_iterations_ -= extent->value;
    for (int64_t i = 0; safe && i < extent->value; ++i) {
      Map<Var, PrimExpr> values;
      values.Set(op->loop_var,
                 IntImm(op->loop_var.dtype(), minimum->value + i));
      VisitStmt(Substitute(op->body, values));
    }
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (op->predicate.defined() && !is_one(op->predicate.value()))
      safe = false;
    Add(&writes, op->buffer, op->indices, Integer(1));
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (op->predicate.defined() && !is_one(op->predicate.value()))
      safe = false;
    Add(&reads, op->buffer, op->indices, Integer(1));
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const VarNode *op) final {
    auto it = allocations_.find(GetRef<Var>(op));
    if (it == allocations_.end())
      return;
    Array<PrimExpr> indices;
    PrimExpr count = Integer(1);
    for (const PrimExpr &dim : it->second->shape) {
      indices.push_back(Integer(0));
      count = count * dim;
    }
    Add(&reads, it->second, indices, count);
  }

  void VisitExpr_(const CallNode *op) final {
    const std::string name = GetOpName(op->op);
    if (IsLdmatrixOp(name)) {
      const bool swizzled = name == "tl.tix_ldmatrix_swzl";
      const bool b8 = name == "tl.tix_ldmatrix_b8";
      const auto *num =
          op->args.size() > 1 ? op->args[1].as<IntImmNode>() : nullptr;
      if (op->args.size() != (swizzled ? 6U : 4U) || !num ||
          (num->value != 1 && num->value != 2 && num->value != 4) ||
          (b8 && num->value != 2) || (swizzled && num->value != 4)) {
        safe = false;
        return;
      }
      Pointer(op->args[3], &writes, num->value * (b8 ? 64 : 32));
      for (size_t i = 0; i < op->args.size(); ++i) {
        if (i != 3)
          VisitExpr(op->args[i]);
      }
      return;
    }
    if (name == "tl.access_ptr")
      Pointer(GetRef<PrimExpr>(op), &reads);
    else
      StmtExprVisitor::VisitExpr_(op);
  }

  const BufferSlotMap &allocations_;
  arith::Analyzer analyzer_;
  int64_t remaining_iterations_{1024};
};

// ---------------------------------------------------------------------------
// Helper: flatten SeqStmt
// ---------------------------------------------------------------------------

static Array<Stmt> FlattenSeq(const Stmt &stmt) {
  if (const auto *seq = stmt.as<SeqStmtNode>()) {
    return seq->seq;
  }
  return {stmt};
}

static Stmt MakeSeq(const Array<Stmt> &stmts) {
  Array<Stmt> filtered;
  for (const auto &s : stmts) {
    if (s.defined())
      filtered.push_back(s);
  }
  if (filtered.empty())
    return Evaluate(IntImm(DataType::Int(32), 0));
  if (filtered.size() == 1)
    return filtered[0];
  return SeqStmt(filtered);
}

// ---------------------------------------------------------------------------
// Helper: Instantiate statements for a concrete ki value + slot routing
// ---------------------------------------------------------------------------

static Stmt Instantiate(const Array<Stmt> &stmts, const Var &loop_var,
                        int64_t k_val, const BufferSlotMap &slot_map) {
  Stmt body = MakeSeq(stmts);
  // Substitute loop variable with constant
  Map<Var, PrimExpr> vmap;
  vmap.Set(loop_var, IntImm(loop_var.dtype(), k_val));
  body = Substitute(body, vmap);
  // Route buffer accesses to slot buffers
  if (!slot_map.empty()) {
    BufferRouteInjector injector(slot_map);
    body = injector(body);
  }
  return body;
}

// ---------------------------------------------------------------------------
// Helper: Create slot buffers for a ping-pong buffer
// ---------------------------------------------------------------------------

static Buffer CreateSlotBuffer(const Buffer &original, int slot_id) {
  std::string suffix = "_" + std::to_string(slot_id);
  // Preserve the original scope via its type_annotation
  Var new_var(original->data->name_hint + suffix,
              original->data->type_annotation);
  return Buffer(new_var, original->dtype, original->shape, original->strides,
                original->elem_offset, original->name + suffix,
                original->data_alignment, original->offset_factor,
                original->buffer_type);
}

// ---------------------------------------------------------------------------
// Helper: Strip annotation from a For loop
// ---------------------------------------------------------------------------

static Stmt StripForAnnotation(const For &op) {
  Map<String, ffi::Any> anns;
  for (const auto &kv : op->annotations) {
    if (kv.first != kRegisterPipelineStage) {
      anns.Set(kv.first, kv.second);
    }
  }
  For loop = op;
  loop.CopyOnWrite()->annotations = std::move(anns);
  return loop;
}

// ---------------------------------------------------------------------------
// Main Rewriter: processes each SBlock independently
// ---------------------------------------------------------------------------

class RegisterPingPongRewriter : public StmtMutator {
public:
  Stmt VisitStmt_(const SBlockNode *op) final {
    // Keep initialization and epilogue statements in place. Multiple pipeline
    // loops sharing one allocation scope need separate lifetime analysis.
    Array<Stmt> block_stmts = FlattenSeq(op->body);
    Optional<For> annotated_loop;
    int pipeline_loops = 0;
    for (const Stmt &stmt : block_stmts) {
      if (auto loop = stmt.as<For>();
          loop && loop.value()->annotations.count(kRegisterPipelineStage)) {
        annotated_loop = loop;
        ++pipeline_loops;
      }
    }

    if (pipeline_loops != 1) {
      // Not our target — recurse normally into body to find nested SBlocks
      Stmt body = VisitStmt(op->body);
      Optional<Stmt> init;
      if (op->init.defined()) {
        init = VisitStmt(op->init.value());
      }
      if (body.same_as(op->body) &&
          (!op->init.defined() || init.value().same_as(op->init.value()))) {
        return GetRef<Stmt>(op);
      }
      return SBlock(op->iter_vars, op->reads, op->writes, op->name_hint, body,
                    init, op->alloc_buffers, op->match_buffers,
                    op->annotations);
    }

    For ann_loop = annotated_loop.value();
    SBlock block = GetRef<SBlock>(op);

    // Get stage from annotation
    auto stage_opt = ann_loop->annotations.Get(kRegisterPipelineStage);
    if (!stage_opt)
      return StripAndReturn(block, ann_loop);
    auto stage_val = GetAnnotationInt(stage_opt.value());
    if (!stage_val.has_value() || stage_val.value() <= 1) {
      return StripAndReturn(block, ann_loop);
    }
    int stage = static_cast<int>(stage_val.value());

    const auto *minimum = ann_loop->min.as<IntImmNode>();
    if (!minimum ||
        (ann_loop->step.defined() && !is_one(ann_loop->step.value())) ||
        (ann_loop->kind != ForKind::kSerial &&
         ann_loop->kind != ForKind::kUnrolled) ||
        op->init.defined() || !op->match_buffers.empty()) {
      return StripAndReturn(block, ann_loop);
    }

    // Get extent
    const auto *extent_imm = ann_loop->extent.as<IntImmNode>();
    if (!extent_imm || extent_imm->value <= 1) {
      return StripAndReturn(block, ann_loop);
    }
    int num_ki = static_cast<int>(extent_imm->value);
    int factor = std::min(stage, num_ki);

    // Collect ping-pong candidates from this loop's body
    LdmatrixDestCollector ld_collector;
    ld_collector(ann_loop->body);
    SharedToLocalLoadCollector stl_collector;
    stl_collector(ann_loop->body);

    VarSet pingpong;
    pingpong.insert(ld_collector.dest_data_vars.begin(),
                    ld_collector.dest_data_vars.end());
    pingpong.insert(stl_collector.dest_data_vars.begin(),
                    stl_collector.dest_data_vars.end());

    if (pingpong.empty())
      return StripAndReturn(block, ann_loop);
    for (const Stmt &stmt : block_stmts) {
      if (!stmt.same_as(ann_loop) && ReferencesAny(stmt, pingpong))
        return StripAndReturn(block, ann_loop);
    }
    for (const auto &region : op->reads) {
      if (pingpong.count(region->buffer->data))
        return StripAndReturn(block, ann_loop);
    }
    for (const auto &region : op->writes) {
      if (pingpong.count(region->buffer->data))
        return StripAndReturn(block, ann_loop);
    }

    // Build slot_buffers from THIS SBlock's alloc_buffers
    std::unordered_map<Var, std::vector<Buffer>, ffi::ObjectPtrHash,
                       ffi::ObjectPtrEqual>
        slot_buffers;
    BufferSlotMap allocations;
    for (const Buffer &buf : op->alloc_buffers) {
      if (!pingpong.count(buf->data))
        continue;
      if (GetBufferScope(buf) != "local")
        return StripAndReturn(block, ann_loop);
      allocations.emplace(buf->data, buf);
      std::vector<Buffer> bufs;
      bufs.reserve(factor);
      for (int slot = 0; slot < factor; slot++) {
        bufs.push_back(CreateSlotBuffer(buf, slot));
      }
      slot_buffers[buf->data] = std::move(bufs);
    }

    // Only proceed if every ping-pong buffer has a matching allocation
    if (slot_buffers.size() != pingpong.size()) {
      return StripAndReturn(block, ann_loop);
    }

    // Split loop body into load / compute groups
    Var loop_var = ann_loop->loop_var;
    Array<Stmt> children = FlattenSeq(ann_loop->body);
    VarSet pp_vars;
    for (const auto &[dvar, _] : slot_buffers) {
      pp_vars.insert(dvar);
    }

    Array<Stmt> load_stmts, compute_stmts;
    PipelineAccesses preceding_computes;
    for (const auto &s : children) {
      if (StmtIsLoad(s, pp_vars)) {
        // Chained RS prepares A in registers before loading B. Move the B load
        // only when earlier computation neither reads nor writes its
        // destination. Source dependencies and side effects are checked for
        // both full groups below, including dependencies across consecutive ki
        // iterations.
        PipelineAccesses load;
        load(s);
        for (const Var &var : load.writes) {
          if (preceding_computes.reads.count(var) ||
              preceding_computes.writes.count(var))
            return StripAndReturn(block, ann_loop);
        }
        load_stmts.push_back(s);
      } else {
        compute_stmts.push_back(s);
        preceding_computes(s);
      }
    }

    if (load_stmts.empty() || compute_stmts.empty()) {
      return StripAndReturn(block, ann_loop);
    }

    PipelineAccesses loads, computes;
    for (const Stmt &stmt : load_stmts)
      loads(stmt);
    for (const Stmt &stmt : compute_stmts)
      computes(stmt);
    if (!loads.safe || !computes.safe)
      return StripAndReturn(block, ann_loop);
    for (const Var &var : loads.writes) {
      if (!pp_vars.count(var))
        return StripAndReturn(block, ann_loop);
    }
    for (const Var &var : loads.reads) {
      if (pp_vars.count(var) || computes.writes.count(var)) {
        return StripAndReturn(block, ann_loop);
      }
    }

    RegisterFootprints definitions(allocations), uses(allocations);
    for (const Stmt &stmt : load_stmts)
      definitions(stmt);
    for (const Stmt &stmt : compute_stmts)
      uses(stmt);
    if (!uses.ReadsCoveredBy(definitions))
      return StripAndReturn(block, ann_loop);

    // Build slot map for a given slot index
    auto build_slot_map = [&slot_buffers](int slot) {
      BufferSlotMap smap;
      for (const auto &[dvar, bufs] : slot_buffers) {
        smap[dvar] = bufs[slot];
      }
      return smap;
    };

    // Fully unroll the pipeline
    Array<Stmt> unrolled;

    // Prologue: ld(0) → slot 0
    unrolled.push_back(
        Instantiate(load_stmts, loop_var, minimum->value, build_slot_map(0)));

    // Steady state: for k in [0, num_ki - 2]
    for (int k = 0; k < num_ki - 1; k++) {
      // ld(k+1) → slot (k+1) % factor
      unrolled.push_back(Instantiate(load_stmts, loop_var,
                                     minimum->value + k + 1,
                                     build_slot_map((k + 1) % factor)));
      // mma(k) → slot k % factor
      unrolled.push_back(Instantiate(compute_stmts, loop_var,
                                     minimum->value + k,
                                     build_slot_map(k % factor)));
    }

    // Epilogue: mma(num_ki - 1) → slot (num_ki - 1) % factor
    unrolled.push_back(Instantiate(compute_stmts, loop_var,
                                   minimum->value + num_ki - 1,
                                   build_slot_map((num_ki - 1) % factor)));

    Stmt new_body = MakeSeq(unrolled);

    // Build new alloc_buffers: replace ping-pong buffers with slot buffers
    Array<Buffer> new_allocs;
    for (const auto &buf : op->alloc_buffers) {
      auto it = slot_buffers.find(buf->data);
      if (it != slot_buffers.end()) {
        for (const auto &slot_buf : it->second) {
          new_allocs.push_back(slot_buf);
        }
      } else {
        new_allocs.push_back(buf);
      }
    }

    return ReplaceLoop(block, ann_loop, new_body, new_allocs);
  }

private:
  static Stmt ReplaceLoop(const SBlock &block, const For &loop,
                          const Stmt &replacement,
                          const Array<Buffer> &alloc_buffers) {
    Array<Stmt> stmts = FlattenSeq(block->body);
    for (size_t i = 0; i < stmts.size(); ++i) {
      if (stmts[i].same_as(loop))
        stmts.Set(i, replacement);
    }
    SBlock result = block;
    auto *node = result.CopyOnWrite();
    node->body = MakeSeq(stmts);
    node->alloc_buffers = alloc_buffers;
    return result;
  }

  // Strip annotation and return the SBlock with original alloc_buffers
  Stmt StripAndReturn(const SBlock &op, const For &loop) {
    return ReplaceLoop(op, loop, StripForAnnotation(loop), op->alloc_buffers);
  }
};

// ---------------------------------------------------------------------------
// Fallback: strip remaining annotations (safety net)
// ---------------------------------------------------------------------------

class StripAnnotationMutator : public StmtMutator {
public:
  Stmt VisitStmt_(const ForNode *op) final {
    Stmt body = VisitStmt(op->body);
    bool has_anno = op->annotations.count(kRegisterPipelineStage);
    if (!has_anno && body.same_as(op->body)) {
      return GetRef<Stmt>(op);
    }
    Map<String, ffi::Any> anns;
    for (const auto &kv : op->annotations) {
      if (kv.first != kRegisterPipelineStage) {
        anns.Set(kv.first, kv.second);
      }
    }
    auto n = CopyOnWrite(op);
    n->body = std::move(body);
    n->annotations = std::move(anns);
    return Stmt(n);
  }
};

// ---------------------------------------------------------------------------
// Main pass implementation
// ---------------------------------------------------------------------------

static PrimFunc RegisterPingPongPipelineImpl(PrimFunc func) {
  bool unsupported_metadata =
      func->attrs.defined() && func->attrs->dict.count(attr::kSafeValueMap);
  PostOrderVisit(func->body, [&](const ObjectRef &node) {
    if (const auto *attribute = node.as<AttrStmtNode>()) {
      unsupported_metadata |= attribute->attr_key == "volatile_scope" ||
                              attribute->attr_key == attr::kSafeValueMap;
    } else if (const auto *allocation = node.as<AllocBufferNode>()) {
      unsupported_metadata |=
          allocation->annotations.count(tirx::attr::kVolatile) != 0;
    } else if (const auto *block = node.as<SBlockNode>()) {
      unsupported_metadata |=
          block->annotations.count(attr::kSafeValueMap) != 0;
    }
  });
  Stmt new_body = func->body;
  if (!unsupported_metadata) {
    RegisterPingPongRewriter rewriter;
    new_body = rewriter(new_body);
  }

  // Safety net: strip any remaining annotations that weren't processed
  StripAnnotationMutator stripper;
  new_body = stripper(new_body);

  if (new_body.same_as(func->body))
    return func;

  auto fptr = func.CopyOnWrite();
  fptr->body = tirx::ConvertSSA(std::move(new_body));
  return func;
}

} // anonymous namespace

// ---------------------------------------------------------------------------
// Pass registration
// ---------------------------------------------------------------------------

Pass RegisterPingPongPipeline_Pass() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return RegisterPingPongPipelineImpl(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ppu.RegisterPingPongPipeline",
                            {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.ppu.transform.RegisterPingPongPipeline",
                        RegisterPingPongPipeline_Pass);
}

} // namespace tl
} // namespace tvm
