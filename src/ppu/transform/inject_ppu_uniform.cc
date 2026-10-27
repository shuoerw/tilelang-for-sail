/*!
 * \file inject_ppu_uniform.cc
 * \brief Preserve legacy PPU uniform-value injection for AIU and swizzled
 *        ldmatrix calls, and additionally rewrite warp-aligned if-guards into
 *        scalar PPU uniform branches.
 */

#include "support/check.h"

#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <unordered_set>
#include <utility>
#include <vector>

#include "ppu/op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

static constexpr int kWarpSize = 32;

/*!
 * \brief Information extracted from a comparison sub-expression suitable for
 *        warp-level predicate rewriting.
 */
struct CompareInfo {
  /*! \brief The "base" expression involving the thread variable. */
  PrimExpr base;
  /*! \brief Right-shift amount already applied (base = orig >> shift_amount).
   */
  int shift_amount = 0;
  /*! \brief The constant RHS of the comparison. */
  int64_t rhs = 0;
  /*! \brief Whether the comparison is LT (true) or GE (false). */
  bool is_lt = true;
};

static bool HasPpuUniform(const PrimExpr &expr) {
  if (const auto *call = expr.as<CallNode>()) {
    if (call->op.same_as(tl::ppu_to_uniform_b32())) {
      return true;
    }
  }
  return false;
}

static bool ContainsPpuUniform(const PrimExpr &expr) {
  bool found = false;
  PostOrderVisit(expr, [&found](const ObjectRef &node) {
    if (found) {
      return;
    }
    if (const auto *call = node.as<CallNode>()) {
      found = call->op.same_as(tl::ppu_to_uniform_b32());
    }
  });
  return found;
}

static PrimExpr MakePpuUniform(const PrimExpr &value) {
  if (HasPpuUniform(value)) {
    return value;
  }
  const DataType dtype = value.dtype();
  PrimExpr value_i32 =
      dtype == DataType::Int(32) ? value : Cast(DataType::Int(32), value);
  PrimExpr uniform =
      Call(DataType::Int(32), tl::ppu_to_uniform_b32(), {value_i32});
  return dtype == DataType::Int(32) ? uniform : Cast(dtype, uniform);
}

/*!
 * \brief Check whether `expr` is (or contains) the given thread_var,
 *        through int32 casts and offsets that preserve warp boundaries.
 */
static bool IsBasedOnThreadVar(const PrimExpr &expr,
                               const VarNode *thread_var) {
  if (const auto *var = expr.as<VarNode>()) {
    return var == thread_var;
  }
  if (const auto *cast = expr.as<CastNode>()) {
    return cast->dtype == DataType::Int(32) &&
           cast->value.dtype() == DataType::Int(32) &&
           IsBasedOnThreadVar(cast->value, thread_var);
  }
  auto aligned_offset = [](const PrimExpr &value) {
    const auto *imm = value.as<IntImmNode>();
    return imm && imm->value % kWarpSize == 0;
  };
  if (const auto *add = expr.as<AddNode>()) {
    return (IsBasedOnThreadVar(add->a, thread_var) && aligned_offset(add->b)) ||
           (IsBasedOnThreadVar(add->b, thread_var) && aligned_offset(add->a));
  }
  if (const auto *sub = expr.as<SubNode>()) {
    return IsBasedOnThreadVar(sub->a, thread_var) && aligned_offset(sub->b);
  }
  return false;
}

/*!
 * \brief Try to decompose `expr` into a form involving thread_var:
 *        base >> shift_amount, where base is one of:
 *        - thread_var
 *        - thread_var & mask
 *        - thread_var % K
 *
 * \return shift_amount >= 0 if matched, -1 otherwise.
 *         On success, `out_base` is set to the base expression.
 */
static int ExtractBaseAndShift(const PrimExpr &expr, const VarNode *thread_var,
                               PrimExpr *out_base) {
  // Case: base >> a  (Call to right_shift op)
  if (const auto *call = expr.as<CallNode>()) {
    static const Op &right_shift_op = Op::Get("tirx.shift_right");
    if (call->op.same_as(right_shift_op) && call->args.size() == 2) {
      if (const auto *imm = call->args[1].as<IntImmNode>()) {
        if (imm->value < 0 || imm->value >= 32)
          return -1;
        int shift = static_cast<int>(imm->value);
        PrimExpr inner_base;
        int inner_shift =
            ExtractBaseAndShift(call->args[0], thread_var, &inner_base);
        if (inner_shift >= 0) {
          *out_base = inner_base;
          return inner_shift + shift;
        }
      }
    }
  }

  // FloorDiv(base, const) where const is power of 2
  if (const auto *fdiv = expr.as<FloorDivNode>()) {
    if (const auto *imm = fdiv->b.as<IntImmNode>()) {
      int64_t divisor = imm->value;
      if (divisor > 0 && (divisor & (divisor - 1)) == 0) {
        // divisor is a power of 2
        int shift = 0;
        int64_t tmp = divisor;
        while (tmp > 1) {
          shift++;
          tmp >>= 1;
        }
        // Recurse on the numerator
        PrimExpr inner_base;
        int inner_shift = ExtractBaseAndShift(fdiv->a, thread_var, &inner_base);
        if (inner_shift >= 0) {
          *out_base = inner_base;
          return inner_shift + shift;
        }
      }
    }
  }

  // Check if expr itself is based on thread_var (shift = 0)
  // Direct thread_var (possibly cast)
  if (IsBasedOnThreadVar(expr, thread_var)) {
    *out_base = expr;
    return 0;
  }

  // thread_var & mask (bitwise and with constant)
  {
    static const Op &bitwise_and_op = Op::Get("tirx.bitwise_and");
    if (const auto *call = expr.as<CallNode>()) {
      if (call->op.same_as(bitwise_and_op) && call->args.size() == 2) {
        if (IsBasedOnThreadVar(call->args[0], thread_var) &&
            call->args[1].as<IntImmNode>()) {
          *out_base = expr;
          return 0;
        }
        if (IsBasedOnThreadVar(call->args[1], thread_var) &&
            call->args[0].as<IntImmNode>()) {
          *out_base = expr;
          return 0;
        }
      }
    }
  }

  // FloorMod(thread_var, K)
  if (const auto *fmod = expr.as<FloorModNode>()) {
    const auto *imm = fmod->b.as<IntImmNode>();
    if (imm && imm->value > 0 && imm->value % kWarpSize == 0 &&
        IsBasedOnThreadVar(fmod->a, thread_var)) {
      *out_base = expr;
      return 0;
    }
  }

  // Mod(thread_var, K)
  if (const auto *mod = expr.as<ModNode>()) {
    const auto *imm = mod->b.as<IntImmNode>();
    if (imm && imm->value > 0 && imm->value % kWarpSize == 0 &&
        IsBasedOnThreadVar(mod->a, thread_var)) {
      *out_base = expr;
      return 0;
    }
  }

  return -1;
}

static bool IsPpuTarget(const PrimFunc &func) {
  auto target = func->GetAttr<Target>(tvm::attr::kTarget);
  return target && target.value()->kind->name == "ppu";
}

/*!
 * \brief Try to match a single comparison expression as a warp-aligned
 *        thread predicate.
 *
 * Matches patterns like:
 *   - threadIdx.x < 128
 *   - (threadIdx.x >> 6) < 4
 *   - (threadIdx.x & 127) < 64
 *   - threadIdx.x >= 64
 *
 * \return true if the pattern matches and the boundary is warp-aligned.
 */
static bool MatchWarpAlignedCompare(const PrimExpr &expr,
                                    const VarNode *thread_var,
                                    CompareInfo *info) {
  const IntImmNode *rhs_imm = nullptr;
  PrimExpr lhs;
  bool is_lt = true;

  if (const auto *lt = expr.as<LTNode>()) {
    lhs = lt->a;
    rhs_imm = lt->b.as<IntImmNode>();
    is_lt = true;
  } else if (const auto *ge = expr.as<GENode>()) {
    lhs = ge->a;
    rhs_imm = ge->b.as<IntImmNode>();
    is_lt = false;
  } else {
    return false;
  }

  if (!rhs_imm || rhs_imm->value <= 0 || lhs.dtype() != DataType::Int(32) ||
      thread_var->dtype != DataType::Int(32))
    return false;

  if (ContainsPpuUniform(lhs))
    return false;

  // Extract base and shift from lhs
  PrimExpr base;
  int shift_amount = ExtractBaseAndShift(lhs, thread_var, &base);
  if (shift_amount < 0 || shift_amount >= 32)
    return false;

  int64_t rhs_val = rhs_imm->value;

  // Check the boundary without overflowing a shifted comparison constant.
  if (shift_amount < 5 && rhs_val % (1LL << (5 - shift_amount)) != 0)
    return false;

  info->base = base;
  info->shift_amount = shift_amount;
  info->rhs = rhs_val;
  info->is_lt = is_lt;
  return true;
}

/*!
 * \brief Rewrite a single comparison expression to use a PPU uniform value.
 *
 * Given a matched CompareInfo, produce the rewritten predicate:
 * - If shift_amount >= 5: ppu_to_uniform_b32(lhs) OP rhs
 *   (lhs is already warp-uniform since it divides by >= 32)
 * - If shift_amount < 5: ppu_to_uniform_b32(base >> 5)
 *   OP (rhs / (32/2^shift_amount))
 */
static PrimExpr RewriteToWarpPredicate(const PrimExpr &original_lhs,
                                       const CompareInfo &info) {
  DataType dtype = DataType::Int(32);
  PrimExpr scalar_expr;
  int64_t new_rhs;

  if (info.shift_amount >= 5) {
    // lhs is already warp-uniform (divided by >= 32)
    // Just wrap the already warp-uniform value.
    PrimExpr lhs_i32 = (original_lhs.dtype() == dtype)
                           ? original_lhs
                           : Cast(dtype, original_lhs);
    scalar_expr = MakePpuUniform(lhs_i32);
    new_rhs = info.rhs;
  } else {
    // Need to shift base to warp granularity
    // new_lhs = base >> 5 (i.e. base / 32)
    // new_rhs = rhs / (32 / 2^shift_amount) = rhs * 2^shift_amount / 32
    int additional_shift = 5 - info.shift_amount;
    PrimExpr base_i32 =
        (info.base.dtype() == dtype) ? info.base : Cast(dtype, info.base);
    PrimExpr warp_expr = FloorDiv(base_i32, make_const(dtype, kWarpSize));
    scalar_expr = MakePpuUniform(warp_expr);
    new_rhs = info.rhs / (1 << additional_shift);
  }

  PrimExpr rhs_const = make_const(dtype, new_rhs);
  if (info.is_lt) {
    return LT(scalar_expr, rhs_const);
  } else {
    return GE(scalar_expr, rhs_const);
  }
}

/*!
 * \brief Flatten an And-tree into a vector of conjuncts.
 */
static void FlattenAnd(const PrimExpr &expr, std::vector<PrimExpr> *out) {
  if (const auto *and_node = expr.as<AndNode>()) {
    FlattenAnd(and_node->a, out);
    FlattenAnd(and_node->b, out);
  } else {
    out->push_back(expr);
  }
}

/*!
 * \brief Rebuild a conjunction from a vector of expressions.
 */
static PrimExpr RebuildAnd(const std::vector<PrimExpr> &parts) {
  ICHECK(!parts.empty());
  PrimExpr result = parts[0];
  for (size_t i = 1; i < parts.size(); i++) {
    result = And(result, parts[i]);
  }
  return result;
}

/*!
 * \brief Try to rewrite a predicate condition. Handles both single comparisons
 *        and conjunctions (And-trees).
 */
static PrimExpr TryRewriteCondition(const PrimExpr &cond,
                                    const VarNode *thread_var) {
  // Flatten the And-tree
  std::vector<PrimExpr> conjuncts;
  FlattenAnd(cond, &conjuncts);

  bool any_changed = false;
  std::vector<PrimExpr> new_conjuncts;
  new_conjuncts.reserve(conjuncts.size());

  for (const auto &part : conjuncts) {
    CompareInfo info;
    if (MatchWarpAlignedCompare(part, thread_var, &info)) {
      // Extract the original lhs for rewriting
      PrimExpr original_lhs;
      if (const auto *lt = part.as<LTNode>()) {
        original_lhs = lt->a;
      } else if (const auto *ge = part.as<GENode>()) {
        original_lhs = ge->a;
      }
      new_conjuncts.push_back(RewriteToWarpPredicate(original_lhs, info));
      any_changed = true;
    } else {
      new_conjuncts.push_back(part);
    }
  }

  if (!any_changed)
    return cond;
  return RebuildAnd(new_conjuncts);
}

/*!
 * \brief The main mutator that injects PPU uniform values.
 */
class UniformWrapMutator : public StmtExprMutator {
public:
  explicit UniformWrapMutator(
      const std::unordered_set<const VarNode *> &thread_vars)
      : thread_vars_(thread_vars) {}

  PrimExpr Rewrite(const PrimExpr &expr) { return VisitExpr(expr); }

  PrimExpr VisitExpr_(const VarNode *op) final {
    if (thread_vars_.count(op)) {
      Var var = GetRef<Var>(op);
      return MakePpuUniform(var);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

private:
  const std::unordered_set<const VarNode *> &thread_vars_;
};

class PpuUniformInjector : public StmtExprMutator {
public:
  PpuUniformInjector() = default;

  static PrimFunc Apply(PrimFunc f) {
    if (!IsPpuTarget(f)) {
      return f;
    }
    PpuUniformInjector mutator;
    // First pass: collect thread variables from AttrStmt
    mutator.CollectThreadVars(f->body);
    // Second pass: rewrite predicates and intrinsic operands
    f.CopyOnWrite()->body = mutator(f->body);
    return f;
  }

private:
  /*! \brief Collect threadIdx variables from thread_extent AttrStmts. */
  void CollectThreadVars(const Stmt &stmt) {
    class Collector : public StmtExprVisitor {
    public:
      void VisitStmt_(const AttrStmtNode *op) final {
        if (op->attr_key == tirx::attr::thread_extent) {
          IterVar iv = Downcast<IterVar>(op->node);
          std::string tag = iv->thread_tag;
          if (tag.rfind("threadIdx", 0) == 0) {
            thread_vars_.insert(iv->var.get());
            const auto *extent = op->value.as<IntImmNode>();
            if (tag == "threadIdx.x" && extent && extent->value > 0 &&
                extent->value % kWarpSize == 0) {
              warp_thread_vars_.insert(iv->var.get());
            }
          }
        }
        StmtExprVisitor::VisitStmt_(op);
      }
      std::unordered_set<const VarNode *> thread_vars_;
      std::unordered_set<const VarNode *> warp_thread_vars_;
    };

    Collector collector;
    collector(stmt);
    thread_vars_ = std::move(collector.thread_vars_);
    warp_thread_vars_ = std::move(collector.warp_thread_vars_);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(tl::ppu_aiu_load()) ||
        op->op.same_as(tl::tix_ldmatrix_swzl())) {
      Array<PrimExpr> new_args;
      new_args.reserve(op->args.size());
      UniformWrapMutator wrapper(thread_vars_);
      for (const PrimExpr &arg : op->args) {
        new_args.push_back(wrapper.Rewrite(arg));
      }
      Call call = GetRef<Call>(op);
      call.CopyOnWrite()->args = new_args;
      return call;
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    Stmt then_case = this->VisitStmt(op->then_case);
    Optional<Stmt> else_case;
    if (op->else_case.defined()) {
      else_case = this->VisitStmt(op->else_case.value());
    }

    PrimExpr new_cond = VisitExpr(op->condition);
    for (const VarNode *tv : warp_thread_vars_) {
      new_cond = TryRewriteCondition(new_cond, tv);
    }

    if (new_cond.same_as(op->condition) && then_case.same_as(op->then_case) &&
        (!op->else_case.defined() ||
         else_case.value().same_as(op->else_case.value()))) {
      return GetRef<Stmt>(op);
    }
    return IfThenElse(new_cond, then_case,
                      else_case.defined() ? else_case.value() : Stmt());
  }

  std::unordered_set<const VarNode *> thread_vars_;
  std::unordered_set<const VarNode *> warp_thread_vars_;
};

using namespace tirx::transform;

tvm::transform::Pass InjectPpuUniform() {
  auto pass_func = [](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return PpuUniformInjector::Apply(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ppu.InjectPpuUniform", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.ppu.transform.InjectPpuUniform", InjectPpuUniform);
}

} // namespace tl
} // namespace tvm
