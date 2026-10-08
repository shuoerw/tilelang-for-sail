/*!
 * \brief Inject commit/wait sync barriers for AIU loads in stage=0 loops
 * \file inject_aiu_sync_barrier.cc
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_set>
#include <vector>

#include "../../op/builtin.h"
#include "ppu/attr.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

// Forward declaration: defined in transform/inject_pipeline.cc.
namespace software_pipeline {
Stmt LowerAsyncCommitWaitAttrs(const Stmt &stmt);
} // namespace software_pipeline

/*!
 * \brief Check if a statement is an async_scope AttrStmt (pipeline-managed).
 * Statements wrapped in async_scope have their commit/wait handled by
 * InjectSoftwarePipeline and should not be processed by this pass.
 */
bool IsAsyncScope(const Stmt &stmt) {
  if (auto *attr = stmt.as<AttrStmtNode>()) {
    return attr->attr_key == s_tir::attr::async_scope;
  }
  return false;
}

/*!
 * \brief Check if a statement contains an aiu_load call.
 */
bool ContainsAIULoad(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (auto *attr = node.as<AttrStmtNode>()) {
      if (attr->attr_key == attr::kAIUSyncOpt) {
        found = true;
      }
    }
  });
  return found;
}

/*!
 * \brief Check if a statement already contains a commit scope, in either the
 * AttrStmt form (async_commit_queue_scope) or the lowered call form
 * (Evaluate(ptx_commit_group())) emitted by InjectSoftwarePipeline.
 */
bool ContainsCommitScope(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (auto *attr = node.as<AttrStmtNode>()) {
      if (attr->attr_key == s_tir::attr::async_commit_queue_scope) {
        found = true;
      }
    } else if (auto *eval = node.as<EvaluateNode>()) {
      if (auto *call = eval->value.as<CallNode>()) {
        if (call->op.same_as(builtin::ptx_commit_group())) {
          found = true;
        }
      }
    }
  });
  return found;
}

/*!
 * \brief Wrap a statement with async_commit_queue_scope (queue_id=0).
 */
Stmt MakeCommit(const Stmt &body) {
  return AttrStmt(make_zero(DataType::Int(32)),
                  s_tir::attr::async_commit_queue_scope,
                  IntImm(DataType::Int(32), 0), body);
}

/*!
 * \brief Wrap a statement with async_wait_queue_scope (queue_id=0,
 * cnt=wait_count).
 * \param body The statement to wrap.
 * \param wait_count The number of in-flight groups to allow (N in wait<N>).
 *        wait_count=0 means wait for ALL pending groups to complete.
 */
Stmt MakeWait(const Stmt &body, int wait_count) {
  auto zero = make_zero(DataType::Int(32));
  return AttrStmt(zero, s_tir::attr::async_wait_queue_scope,
                  IntImm(DataType::Int(32), 0),
                  AttrStmt(zero, s_tir::attr::async_wait_inflight_count,
                           IntImm(DataType::Int(32), wait_count), body));
}

/*!
 * \brief A pending commit group containing the destinations of one synchronous
 * copy. A copy may emit several AIU loads that share a commit.
 */
struct PendingGroup {
  std::unordered_set<const VarNode *> dst_vars;
};

class AIUSyncBarrierInjector : public StmtExprMutator {
public:
  Stmt VisitStmt_(const ForNode *op) final {
    // Recurse into the loop body with in_for_body_ flag set,
    // so SeqStmt handler can do wrap-around for loop-carried dependencies.
    bool old = in_for_body_;
    in_for_body_ = true;
    Stmt result = StmtExprMutator::VisitStmt_(op);
    in_for_body_ = old;
    return result;
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    Array<Stmt> seq;
    for (const Stmt &s : op->seq) {
      seq.push_back(VisitStmt(s));
    }

    // Left-to-right scan: track pending commit groups in FIFO order.
    // At each use point, compute the minimal wait<N> that guarantees all
    // needed groups have completed while allowing newer ones to stay in-flight.
    std::vector<bool> needs_commit(seq.size(), false);
    std::vector<int> needs_wait(seq.size(), -1);
    std::vector<PendingGroup> pending_groups;
    bool has_any_aiu = false;

    for (int i = 0; i < static_cast<int>(seq.size()); ++i) {
      bool is_container = seq[i].as<ForNode>() ||
                          seq[i].as<SBlockRealizeNode>() ||
                          seq[i].as<SBlockNode>();
      if (!pending_groups.empty()) {
        // For container nodes (For/Block): skip use-detection only if the
        // container also loads the same buffer (prefetch pattern), in which
        // case the internal wrap-around handles the wait. Otherwise the
        // outer wait is needed for correctness.
        bool skip_use =
            is_container && ContainsAIULoadForVars(seq[i], pending_groups);
        if (!skip_use) {
          int newest_needed = FindNewestUsedGroup(seq[i], pending_groups);
          if (newest_needed >= 0) {
            int wait_count =
                static_cast<int>(pending_groups.size()) - newest_needed - 1;
            if (needs_wait[i] < 0 || wait_count < needs_wait[i]) {
              needs_wait[i] = wait_count;
            }
            pending_groups.erase(pending_groups.begin(),
                                 pending_groups.begin() + newest_needed + 1);
          }
        }
      }
      if (ContainsAIULoad(seq[i]) && !is_container && !IsAsyncScope(seq[i])) {
        has_any_aiu = true;
        if (!ContainsCommitScope(seq[i])) {
          needs_commit[i] = true;
        }
        CollectAIUDstGroups(seq[i], &pending_groups);
      }
    }

    if (!has_any_aiu) {
      return SeqStmt(seq);
    }

    // Wrap-around for loop-carried dependencies: pending groups at loop tail
    // are consumed at the beginning of the next iteration.
    // Account for new commits issued BEFORE the use point in the same
    // iteration: at runtime the async queue contains both the carried-over
    // groups (oldest) and the freshly committed groups, so wait<N> must reflect
    // the total.
    if (!pending_groups.empty() && in_for_body_) {
      int new_commits_before = 0;
      for (int i = 0; i < static_cast<int>(seq.size()); ++i) {
        int newest_needed = FindNewestUsedGroup(seq[i], pending_groups);
        if (newest_needed >= 0) {
          int total_inflight =
              static_cast<int>(pending_groups.size()) + new_commits_before;
          int wait_count = total_inflight - newest_needed - 1;
          if (needs_wait[i] < 0 || wait_count < needs_wait[i]) {
            needs_wait[i] = wait_count;
          }
          break;
        }
        // Count commit groups emitted before this point in the iteration.
        if (ContainsAIULoad(seq[i]) && !seq[i].as<ForNode>() &&
            !seq[i].as<SBlockRealizeNode>() && !seq[i].as<SBlockNode>() &&
            !IsAsyncScope(seq[i])) {
          new_commits_before += CountAIULoads(seq[i]);
        }
      }
    }

    // Build result with commits and waits applied.
    Array<Stmt> new_seq;
    for (int i = 0; i < static_cast<int>(seq.size()); ++i) {
      Stmt s = seq[i];
      if (needs_commit[i]) {
        s = InjectCommitInto(s);
      }
      if (needs_wait[i] >= 0) {
        s = MakeWait(s, needs_wait[i]);
      }
      new_seq.push_back(s);
    }

    return SeqStmt(std::move(new_seq));
  }

private:
  bool in_for_body_ = false;

  /*!
   * \brief Find the newest (highest-index) pending group whose dst vars
   * are referenced by the given statement. Returns -1 if none match.
   */
  int FindNewestUsedGroup(const Stmt &stmt,
                          const std::vector<PendingGroup> &groups) const {
    int newest = -1;
    for (int g = 0; g < static_cast<int>(groups.size()); ++g) {
      if (UsesAnyVar(stmt, groups[g].dst_vars)) {
        newest = g;
      }
    }
    return newest;
  }

  /*!
   * \brief Check if a statement contains aiu_load that writes to any of the
   * pending groups' dst vars. Used to detect prefetch patterns inside loops.
   */
  bool ContainsAIULoadForVars(const Stmt &stmt,
                              const std::vector<PendingGroup> &groups) const {
    if (!ContainsAIULoad(stmt)) {
      return false;
    }
    std::unordered_set<const VarNode *> all_vars;
    for (const auto &g : groups) {
      all_vars.insert(g.dst_vars.begin(), g.dst_vars.end());
    }
    bool found = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (found)
        return;
      if (auto *call = node.as<CallNode>()) {
        if (call->op.same_as(tl::ppu_aiu_load()) && call->args.size() > 0) {
          if (auto *access_call = call->args[0].as<CallNode>()) {
            if (access_call->op.same_as(builtin::tvm_access_ptr()) &&
                access_call->args.size() > 1) {
              if (auto *var = access_call->args[1].as<VarNode>()) {
                if (all_vars.count(var)) {
                  found = true;
                }
              }
            }
          }
        }
      }
    });
    return found;
  }

  /*!
   * \brief Count the commit groups emitted by a marked synchronous copy.
   */
  int CountAIULoads(const Stmt &stmt) { return ContainsAIULoad(stmt) ? 1 : 0; }

  /*!
   * \brief Collect the destinations of a marked copy into one pending commit
   * group. arg[0] is tvm_access_ptr(type, buffer_data_var, offset, extent,
   * mask). We extract buffer_data_var (arg[1] of tvm_access_ptr).
   */
  void CollectAIUDstGroups(const Stmt &stmt,
                           std::vector<PendingGroup> *groups) {
    if (!ContainsAIULoad(stmt)) {
      return;
    }
    PendingGroup group;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (auto *call = node.as<CallNode>()) {
        if (call->op.same_as(tl::ppu_aiu_load()) && call->args.size() > 0) {
          PrimExpr dst = call->args[0];
          // dst should be tvm_access_ptr(type, buffer_data, offset, extent,
          // mask)
          if (auto *access_call = dst.as<CallNode>()) {
            if (access_call->op.same_as(builtin::tvm_access_ptr()) &&
                access_call->args.size() > 1) {
              if (auto *var = access_call->args[1].as<VarNode>()) {
                group.dst_vars.insert(var);
              }
            }
          }
        }
      }
    });
    if (!group.dst_vars.empty()) {
      groups->push_back(std::move(group));
    }
  }

  /*!
   * \brief Check if a statement uses any buffer whose data Var is in the set.
   * Checks BufferLoad, BufferStore, and tvm_access_ptr references.
   */
  bool UsesAnyVar(const Stmt &stmt,
                  const std::unordered_set<const VarNode *> &vars) const {
    bool found = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (found)
        return;
      if (auto *load = node.as<BufferLoadNode>()) {
        if (vars.count(load->buffer->data.get())) {
          found = true;
        }
      } else if (auto *store = node.as<BufferStoreNode>()) {
        if (vars.count(store->buffer->data.get())) {
          found = true;
        }
      } else if (auto *call = node.as<CallNode>()) {
        // tvm_access_ptr(type, buffer_data, offset, extent, mask)
        if (call->op.same_as(builtin::tvm_access_ptr()) &&
            call->args.size() > 1) {
          if (auto *var = call->args[1].as<VarNode>()) {
            if (vars.count(var)) {
              found = true;
            }
          }
        }
      }
    });
    return found;
  }

  /*!
   * \brief Inject commit into a statement containing aiu_load.
   * For IfThenElse: push commit into branches to avoid phantom commits.
   * For other stmts: wrap with commit scope.
   */
  Stmt InjectCommitInto(const Stmt &stmt) {
    if (auto *attr = stmt.as<AttrStmtNode>()) {
      if (attr->attr_key == attr::kAIUSyncOpt) {
        return AttrStmt(attr->node, attr->attr_key, attr->value,
                        InjectCommitInto(attr->body));
      }
    }
    if (auto *if_node = stmt.as<IfThenElseNode>()) {
      Stmt then_branch = if_node->then_case;
      if (ContainsAIULoad(then_branch) && !ContainsCommitScope(then_branch)) {
        then_branch = MakeCommit(then_branch);
      }
      Optional<Stmt> else_branch = if_node->else_case;
      if (else_branch && ContainsAIULoad(else_branch.value()) &&
          !ContainsCommitScope(else_branch.value())) {
        else_branch = MakeCommit(else_branch.value());
      }
      return IfThenElse(if_node->condition, then_branch, else_branch);
    }
    // Default: wrap the whole statement with commit
    return MakeCommit(stmt);
  }
};

class AIUSyncOptAnnotationStripper : public StmtExprMutator {
public:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == attr::kAIUSyncOpt) {
      return VisitStmt(op->body);
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

using namespace tirx::transform;

tvm::transform::Pass InjectAIUSyncBarrier() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    auto *n = f.CopyOnWrite();
    n->body = AIUSyncBarrierInjector()(n->body);
    // Only lower annotations to Call nodes when this pass actually inserted
    // commit scopes (stage=0 path). For stage>0, InjectSoftwarePipeline already
    // lowered its own annotations internally.
    const bool has_commit_scope = ContainsCommitScope(n->body);
    if (has_commit_scope) {
      n->body = software_pipeline::LowerAsyncCommitWaitAttrs(n->body);
    }
    n->body = AIUSyncOptAnnotationStripper()(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.InjectAIUSyncBarrier", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.ppu.transform.InjectAIUSyncBarrier",
                        InjectAIUSyncBarrier);
}

} // namespace tl
} // namespace tvm
