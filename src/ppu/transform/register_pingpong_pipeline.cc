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

#include <tvm/ffi/memory.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

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
  std::unordered_set<const VarNode *> dest_data_vars;

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
        dest_data_vars.insert(load->buffer->data.get());
      }
    });
  }
};

// ---------------------------------------------------------------------------
// 2. SharedToLocalLoadCollector
// ---------------------------------------------------------------------------

class SharedToLocalLoadCollector : public StmtExprVisitor {
public:
  std::unordered_set<const VarNode *> dest_data_vars;

  void VisitStmt_(const BufferStoreNode *op) final {
    std::string scope = GetBufferScope(op->buffer);
    if (scope == "local" && ExprReadsNonLocal(op->value)) {
      dest_data_vars.insert(op->buffer->data.get());
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
  explicit BufferRouteInjector(
      const std::unordered_map<const VarNode *, Buffer> &slot_map)
      : slot_map_(slot_map) {}

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    Array<PrimExpr> indices;
    for (const auto &idx : op->indices) {
      indices.push_back(VisitExpr(idx));
    }
    auto it = slot_map_.find(op->buffer->data.get());
    if (it != slot_map_.end()) {
      return BufferLoad(it->second, indices);
    }
    return BufferLoad(op->buffer, indices);
  }

  Stmt VisitStmt_(const BufferStoreNode *op) final {
    PrimExpr value = VisitExpr(op->value);
    Array<PrimExpr> indices;
    for (const auto &idx : op->indices) {
      indices.push_back(VisitExpr(idx));
    }
    auto it = slot_map_.find(op->buffer->data.get());
    if (it != slot_map_.end()) {
      return BufferStore(it->second, value, indices);
    }
    return BufferStore(op->buffer, value, indices);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    Array<PrimExpr> args;
    for (const auto &a : op->args) {
      args.push_back(VisitExpr(a));
    }
    std::string name = GetOpName(op->op);
    if (IsMmaOp(name)) {
      // ptx_mma passes buf.data as an argument; replace by slot buffer's data
      for (size_t idx = 0; idx < op->args.size(); idx++) {
        const auto *var_node = op->args[idx].as<VarNode>();
        if (var_node) {
          auto it = slot_map_.find(var_node);
          if (it != slot_map_.end()) {
            args.Set(idx, it->second->data);
          }
        }
      }
    }
    return Call(op->dtype, op->op, args, op->annotations, op->span);
  }

private:
  const std::unordered_map<const VarNode *, Buffer> &slot_map_;
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
static bool
StmtWritesAny(const Stmt &stmt,
              const std::unordered_set<const VarNode *> &data_vars) {
  bool found = false;
  PostOrderVisit(stmt, [&found, &data_vars](const ObjectRef &obj) {
    if (found)
      return;
    if (const auto *store = obj.as<BufferStoreNode>()) {
      if (data_vars.count(store->buffer->data.get())) {
        found = true;
      }
    }
  });
  return found;
}

// A statement is a "load" if it issues ldmatrix OR stores into a ping-pong
// buffer
static bool
StmtIsLoad(const Stmt &stmt,
           const std::unordered_set<const VarNode *> &pingpong_vars) {
  return StmtContainsLdmatrixOp(stmt) || StmtWritesAny(stmt, pingpong_vars);
}

// Only reorder understood memory operations. Unknown calls, barriers, and
// conditional definitions cannot establish an independent next-iteration load.
class PipelineAccesses : public StmtExprVisitor {
public:
  bool safe = true;
  std::unordered_set<const VarNode *> reads, writes;

  void VisitExpr_(const BufferLoadNode *op) final {
    reads.insert(op->buffer->data.get());
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    writes.insert(op->buffer->data.get());
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfThenElseNode *) final { safe = false; }
  void VisitStmt_(const BindNode *) final { safe = false; }
  void VisitStmt_(const SBlockNode *) final { safe = false; }
  void VisitStmt_(const AttrStmtNode *) final { safe = false; }

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
          writes.insert(load->buffer->data.get());
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
        if (const auto *var = arg.as<VarNode>();
            var && var->dtype.is_handle()) {
          reads.insert(var);
          writes.insert(var);
        }
      }
    } else if (name != "tl.access_ptr" &&
               SideEffect(GetRef<Call>(op)) > CallEffectKind::kPure) {
      safe = false;
    }
    StmtExprVisitor::VisitExpr_(op);
  }
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

static Stmt
Instantiate(const Array<Stmt> &stmts, const Var &loop_var, int64_t k_val,
            const std::unordered_map<const VarNode *, Buffer> &slot_map) {
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

static Stmt StripForAnnotation(const ForNode *op) {
  Map<String, ffi::Any> anns;
  for (const auto &kv : op->annotations) {
    if (kv.first != kRegisterPipelineStage) {
      anns.Set(kv.first, kv.second);
    }
  }
  auto n = ffi::make_object<ForNode>(*op);
  n->annotations = std::move(anns);
  return Stmt(n);
}

// ---------------------------------------------------------------------------
// Main Rewriter: processes each SBlock independently
// ---------------------------------------------------------------------------

class RegisterPingPongRewriter : public StmtMutator {
public:
  Stmt VisitStmt_(const SBlockNode *op) final {
    // Check if body is directly an annotated For loop
    const ForNode *ann_loop = op->body.as<ForNode>();
    bool body_is_annotated =
        ann_loop && ann_loop->annotations.count(kRegisterPipelineStage);

    if (!body_is_annotated) {
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

    // === This SBlock's body IS an annotated For loop — process locally ===

    // Get stage from annotation
    auto stage_opt = ann_loop->annotations.Get(kRegisterPipelineStage);
    if (!stage_opt)
      return StripAndReturn(op, ann_loop);
    auto stage_val = GetAnnotationInt(stage_opt.value());
    if (!stage_val.has_value() || stage_val.value() <= 1) {
      return StripAndReturn(op, ann_loop);
    }
    int stage = static_cast<int>(stage_val.value());

    const auto *minimum = ann_loop->min.as<IntImmNode>();
    if (!minimum ||
        (ann_loop->step.defined() && !is_one(ann_loop->step.value())) ||
        (ann_loop->kind != ForKind::kSerial &&
         ann_loop->kind != ForKind::kUnrolled) ||
        op->init.defined() || !op->match_buffers.empty()) {
      return StripAndReturn(op, ann_loop);
    }

    // Get extent
    const auto *extent_imm = ann_loop->extent.as<IntImmNode>();
    if (!extent_imm || extent_imm->value <= 1) {
      return StripAndReturn(op, ann_loop);
    }
    int num_ki = static_cast<int>(extent_imm->value);
    int factor = std::min(stage, num_ki);

    // Collect ping-pong candidates from this loop's body
    LdmatrixDestCollector ld_collector;
    ld_collector(ann_loop->body);
    SharedToLocalLoadCollector stl_collector;
    stl_collector(ann_loop->body);

    std::unordered_set<const VarNode *> pingpong;
    pingpong.insert(ld_collector.dest_data_vars.begin(),
                    ld_collector.dest_data_vars.end());
    pingpong.insert(stl_collector.dest_data_vars.begin(),
                    stl_collector.dest_data_vars.end());

    if (pingpong.empty())
      return StripAndReturn(op, ann_loop);
    for (const auto &region : op->reads) {
      if (pingpong.count(region->buffer->data.get()))
        return StripAndReturn(op, ann_loop);
    }
    for (const auto &region : op->writes) {
      if (pingpong.count(region->buffer->data.get()))
        return StripAndReturn(op, ann_loop);
    }

    // Build slot_buffers from THIS SBlock's alloc_buffers
    std::unordered_map<const VarNode *, std::vector<Buffer>> slot_buffers;
    for (const Buffer &buf : op->alloc_buffers) {
      if (!pingpong.count(buf->data.get()))
        continue;
      std::vector<Buffer> bufs;
      bufs.reserve(factor);
      for (int slot = 0; slot < factor; slot++) {
        bufs.push_back(CreateSlotBuffer(buf, slot));
      }
      slot_buffers[buf->data.get()] = std::move(bufs);
    }

    // Only proceed if every ping-pong buffer has a matching allocation
    if (slot_buffers.size() != pingpong.size()) {
      return StripAndReturn(op, ann_loop);
    }

    // Split loop body into load / compute groups
    Var loop_var = ann_loop->loop_var;
    Array<Stmt> children = FlattenSeq(ann_loop->body);
    std::unordered_set<const VarNode *> pp_vars;
    for (const auto &[dvar, _] : slot_buffers) {
      pp_vars.insert(dvar);
    }

    Array<Stmt> load_stmts, compute_stmts;
    for (const auto &s : children) {
      if (StmtIsLoad(s, pp_vars)) {
        if (!compute_stmts.empty())
          return StripAndReturn(op, ann_loop);
        load_stmts.push_back(s);
      } else {
        compute_stmts.push_back(s);
      }
    }

    if (load_stmts.empty() || compute_stmts.empty()) {
      return StripAndReturn(op, ann_loop);
    }

    PipelineAccesses loads, computes;
    for (const Stmt &stmt : load_stmts)
      loads(stmt);
    for (const Stmt &stmt : compute_stmts)
      computes(stmt);
    if (!loads.safe || !computes.safe)
      return StripAndReturn(op, ann_loop);
    for (const VarNode *var : loads.writes) {
      if (!pp_vars.count(var))
        return StripAndReturn(op, ann_loop);
    }
    for (const VarNode *var : loads.reads) {
      if (pp_vars.count(var) || computes.writes.count(var)) {
        return StripAndReturn(op, ann_loop);
      }
    }

    // Build slot map for a given slot index
    auto build_slot_map = [&slot_buffers](int slot) {
      std::unordered_map<const VarNode *, Buffer> smap;
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
      auto it = slot_buffers.find(buf->data.get());
      if (it != slot_buffers.end()) {
        for (const auto &slot_buf : it->second) {
          new_allocs.push_back(slot_buf);
        }
      } else {
        new_allocs.push_back(buf);
      }
    }

    Optional<Stmt> init;
    if (op->init.defined())
      init = op->init.value();

    return SBlock(op->iter_vars, op->reads, op->writes, op->name_hint, new_body,
                  init, new_allocs, op->match_buffers, op->annotations);
  }

private:
  // Strip annotation and return the SBlock with original alloc_buffers
  Stmt StripAndReturn(const SBlockNode *op, const ForNode *loop) {
    Stmt stripped_loop = StripForAnnotation(loop);
    Optional<Stmt> init;
    if (op->init.defined())
      init = op->init.value();
    return SBlock(op->iter_vars, op->reads, op->writes, op->name_hint,
                  stripped_loop, init, op->alloc_buffers, op->match_buffers,
                  op->annotations);
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
  // Apply the per-SBlock rewriter
  RegisterPingPongRewriter rewriter;
  Stmt new_body = rewriter(func->body);

  // Safety net: strip any remaining annotations that weren't processed
  StripAnnotationMutator stripper;
  new_body = stripper(new_body);

  if (new_body.same_as(func->body))
    return func;

  auto fptr = func.CopyOnWrite();
  fptr->body = std::move(new_body);
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
