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

#include <algorithm>
#include <unordered_set>
#include <vector>

#include "../../op/builtin.h"
#include "ppu/op/copy.h"

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
    if (auto *call = node.as<CallNode>()) {
      if (ppu::IsCompilerSyncAiuLoad(call)) {
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
 * \brief A pending commit group, or a conservative batch across branches.
 * Entries retain FIFO order; conditional batches track a lower bound on
 * commits.
 */
struct PendingGroup {
  std::unordered_set<const VarNode *> dst_vars;
  // A conditional batch may issue different numbers of groups on each path.
  // Only its minimum count is safe to allow to remain in flight.
  int guaranteed_commits = 1;
};

// A commit closes every uncommitted load on queue 0. Compiler-managed loads
// cannot be synchronized independently while a caller-managed group is open.
class AIUAsyncGroupValidator : public StmtExprVisitor {
private:
  void VisitExpr_(const CallNode *op) final {
    StmtExprVisitor::VisitExpr_(op);
    if (op->op.same_as(tl::ppu_aiu_load())) {
      if (ppu::IsCompilerSyncAiuLoad(op)) {
        ICHECK(!uncommitted_caller_load_)
            << "Cannot insert compiler-managed AIU synchronization into an "
               "uncommitted caller-managed async group. Call "
               "T.ptx_commit_group() before a compiler-managed AIU load "
               "(including T.copy), or use caller-managed loads for the "
               "entire group.";
      } else {
        uncommitted_caller_load_ = true;
      }
    } else if (op->op.same_as(builtin::ptx_commit_group())) {
      uncommitted_caller_load_ = false;
    }
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    VisitExpr(op->condition);
    if (const int64_t *condition = as_const_int(op->condition)) {
      if (*condition) {
        VisitStmt(op->then_case);
      } else if (op->else_case.defined()) {
        VisitStmt(op->else_case.value());
      }
      return;
    }
    bool incoming = uncommitted_caller_load_;
    VisitStmt(op->then_case);
    bool then_pending = uncommitted_caller_load_;
    uncommitted_caller_load_ = incoming;
    if (op->else_case.defined()) {
      VisitStmt(op->else_case.value());
    }
    uncommitted_caller_load_ |= then_pending;
  }

  void VisitLoop(const Stmt &body, bool at_least_once, bool exactly_once) {
    bool incoming = uncommitted_caller_load_;
    VisitStmt(body);
    bool outgoing = uncommitted_caller_load_;
    // A group opened at the tail can reach a compiler load in the next
    // iteration. The boolean may-state reaches a fixed point in two visits.
    if (!exactly_once && !incoming && outgoing) {
      VisitStmt(body);
      outgoing |= uncommitted_caller_load_;
    }
    uncommitted_caller_load_ = outgoing || (!at_least_once && incoming);
  }

  void VisitStmt_(const ForNode *op) final {
    VisitExpr(op->min);
    VisitExpr(op->extent);
    const int64_t *extent = as_const_int(op->extent);
    if (extent && *extent <= 0) {
      return;
    }
    VisitLoop(op->body, extent && *extent > 0, extent && *extent == 1);
  }

  void VisitStmt_(const WhileNode *op) final {
    VisitExpr(op->condition);
    const int64_t *condition = as_const_int(op->condition);
    if (condition && *condition == 0) {
      return;
    }
    VisitLoop(op->body, false, false);
  }

  void VisitStmt_(const AttrStmtNode *op) final {
    StmtExprVisitor::VisitStmt_(op);
    if (op->attr_key == s_tir::attr::async_commit_queue_scope) {
      const int64_t *queue = as_const_int(op->value);
      if (queue && *queue == 0) {
        uncommitted_caller_load_ = false;
      }
    }
  }

  bool uncommitted_caller_load_ = false;
};

class AIUSyncBarrierInjector : public StmtExprMutator {
public:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == s_tir::attr::async_scope ||
        op->attr_key == s_tir::attr::async_commit_queue_scope) {
      // The surrounding scope already owns the group's commit boundary.
      return GetRef<Stmt>(op);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    // A commit in one branch must not suppress insertion in another branch.
    return InjectCommitInto(StmtExprMutator::VisitStmt_(op));
  }

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
                CountGuaranteedCommits(pending_groups, newest_needed + 1);
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
          int wait_count =
              CountGuaranteedCommits(pending_groups, newest_needed + 1) +
              new_commits_before;
          if (needs_wait[i] < 0 || wait_count < needs_wait[i]) {
            needs_wait[i] = wait_count;
          }
          break;
        }
        // Count commit groups emitted before this point in the iteration.
        if (ContainsAIULoad(seq[i]) && !seq[i].as<ForNode>() &&
            !seq[i].as<SBlockRealizeNode>() && !seq[i].as<SBlockNode>() &&
            !IsAsyncScope(seq[i])) {
          new_commits_before += CountGuaranteedCommitsIn(seq[i]);
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
    std::unordered_set<const VarNode *> all_vars;
    for (const auto &g : groups) {
      all_vars.insert(g.dst_vars.begin(), g.dst_vars.end());
    }
    bool found = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (found)
        return;
      if (auto *call = node.as<CallNode>()) {
        if (ppu::IsCompilerSyncAiuLoad(call) && call->args.size() > 0) {
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

  static int CountGuaranteedCommits(const std::vector<PendingGroup> &groups,
                                    int begin = 0) {
    int count = 0;
    for (int i = begin; i < static_cast<int>(groups.size()); ++i) {
      count += groups[i].guaranteed_commits;
    }
    return count;
  }

  int CountGuaranteedCommitsIn(const Stmt &stmt) {
    std::vector<PendingGroup> groups;
    CollectAIUDstGroups(stmt, &groups);
    return CountGuaranteedCommits(groups);
  }

  /*!
   * \brief Collect groups in program order, respecting commit scopes.
   * Mutually exclusive branches form one batch with the minimum commit count
   * across paths. Waiting for any destination in that batch waits for the
   * whole batch; only guaranteed later commits contribute to wait<N>.
   * Destinations are extracted from tvm_access_ptr's buffer-data argument.
   */
  void CollectAIUDstGroups(const Stmt &stmt,
                           std::vector<PendingGroup> *groups) {
    if (const auto *branch = stmt.as<IfThenElseNode>()) {
      std::vector<PendingGroup> then_groups, else_groups;
      CollectAIUDstGroups(branch->then_case, &then_groups);
      if (branch->else_case.defined()) {
        CollectAIUDstGroups(branch->else_case.value(), &else_groups);
      }
      PendingGroup batch;
      batch.guaranteed_commits = std::min(CountGuaranteedCommits(then_groups),
                                          CountGuaranteedCommits(else_groups));
      for (const auto &group : then_groups) {
        batch.dst_vars.insert(group.dst_vars.begin(), group.dst_vars.end());
      }
      for (const auto &group : else_groups) {
        batch.dst_vars.insert(group.dst_vars.begin(), group.dst_vars.end());
      }
      if (!batch.dst_vars.empty()) {
        groups->push_back(std::move(batch));
      }
      return;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const auto &child : seq->seq) {
        CollectAIUDstGroups(child, groups);
      }
      return;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      std::vector<PendingGroup> body_groups;
      CollectAIUDstGroups(attr->body, &body_groups);
      if (attr->attr_key == s_tir::attr::async_commit_queue_scope) {
        PendingGroup group;
        for (const auto &body_group : body_groups) {
          group.dst_vars.insert(body_group.dst_vars.begin(),
                                body_group.dst_vars.end());
        }
        if (!group.dst_vars.empty()) {
          groups->push_back(std::move(group));
        }
      } else {
        groups->insert(groups->end(), body_groups.begin(), body_groups.end());
      }
      return;
    }
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (auto *call = node.as<CallNode>()) {
        if (ppu::IsCompilerSyncAiuLoad(call) && call->args.size() > 0) {
          PendingGroup group;
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
          if (!group.dst_vars.empty()) {
            groups->push_back(std::move(group));
          }
        }
      }
    });
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
    if (auto *if_node = stmt.as<IfThenElseNode>()) {
      Stmt then_branch = if_node->then_case;
      if (ContainsAIULoad(then_branch) && !ContainsCommitScope(then_branch)) {
        then_branch = InjectCommitInto(then_branch);
      }
      Optional<Stmt> else_branch = if_node->else_case;
      if (else_branch && ContainsAIULoad(else_branch.value()) &&
          !ContainsCommitScope(else_branch.value())) {
        else_branch = InjectCommitInto(else_branch.value());
      }
      return IfThenElse(if_node->condition, then_branch, else_branch);
    }
    // Default: wrap the whole statement with commit
    return MakeCommit(stmt);
  }
};

using namespace tirx::transform;

tvm::transform::Pass InjectAIUSyncBarrier() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    AIUAsyncGroupValidator()(f->body);
    auto *n = f.CopyOnWrite();
    n->body = AIUSyncBarrierInjector()(n->body);
    // Only lower annotations to Call nodes when this pass actually inserted
    // commit scopes (stage=0 path). For stage>0, InjectSoftwarePipeline already
    // lowered its own annotations internally.
    if (ContainsCommitScope(n->body)) {
      n->body = software_pipeline::LowerAsyncCommitWaitAttrs(n->body);
    }
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
