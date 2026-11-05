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

#include "ppu/op/builtin.h"
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

// Only inspect loads whose synchronization is visible to this pass. Pipeline
// scopes are opaque, even when nested below a branch or another container.
class CompilerAIULoadCollector : public StmtExprVisitor {
public:
  explicit CompilerAIULoadCollector(bool conditional_only = false)
      : conditional_only_(conditional_only) {}

  Array<Call> Collect(const Stmt &stmt) {
    VisitStmt(stmt);
    return loads_;
  }

private:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != s_tir::attr::async_scope) {
      StmtExprVisitor::VisitStmt_(op);
    }
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    bool old = in_branch_;
    in_branch_ = true;
    StmtExprVisitor::VisitStmt_(op);
    in_branch_ = old;
  }

  void VisitExpr_(const CallNode *op) final {
    if ((!conditional_only_ || in_branch_) && ppu::IsCompilerSyncAiuLoad(op)) {
      loads_.push_back(GetRef<Call>(op));
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  const bool conditional_only_;
  bool in_branch_ = false;
  Array<Call> loads_;
};

bool ContainsAIULoad(const Stmt &stmt) {
  return !CompilerAIULoadCollector().Collect(stmt).empty();
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
 * Entries retain FIFO order. Conditional batches only track destinations:
 * functions containing conditional AIU loads always wait for all groups.
 */
struct PendingGroup {
  std::unordered_set<const VarNode *> dst_vars;
  // Container exits summarize destinations, not a fixed number of commits.
  bool unknown_count = false;
};

class AIUSyncBarrierInjector : public StmtExprMutator {
public:
  explicit AIUSyncBarrierInjector(bool wait_all) : wait_all_(wait_all) {}

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
      // An explicit standalone wait-all also retires tracked groups. Do not
      // apply this to a whole container: it can issue new groups after a wait.
      if (IsWaitAll(seq[i])) {
        pending_groups.clear();
      }
      if (!pending_groups.empty()) {
        // Child statements have already been transformed. An existing wait<0>
        // may cover incoming groups, but a reload (or a later wait) does not.
        // Only omit the entry wait when every path to a conflicting access
        // drains the incoming groups first. Keep backedge analysis separate.
        int newest_needed = -1;
        for (int g = 0; g < static_cast<int>(pending_groups.size()); ++g) {
          if (CheckIncomingGroup(seq[i], pending_groups[g].dst_vars) ==
              IncomingState::kConflict) {
            newest_needed = g;
          }
        }
        if (newest_needed >= 0) {
          int wait_count =
              (wait_all_ || HasUnknownCount(pending_groups))
                  ? 0
                  : static_cast<int>(pending_groups.size()) - newest_needed - 1;
          if (needs_wait[i] < 0 || wait_count < needs_wait[i]) {
            needs_wait[i] = wait_count;
          }
          if (wait_count == 0) {
            pending_groups.clear();
          } else {
            pending_groups.erase(pending_groups.begin(),
                                 pending_groups.begin() + newest_needed + 1);
          }
        }
      }
      if (is_container && ContainsAIULoad(seq[i])) {
        // Incoming coverage and outgoing work are different generations.
        // A wait in the loop body can retire the former and then issue the
        // latter. Carry that outgoing may-state to subsequent consumers.
        PendingGroup outgoing;
        outgoing.unknown_count = true;
        for (const auto &group : pending_groups) {
          outgoing.dst_vars.insert(group.dst_vars.begin(),
                                   group.dst_vars.end());
        }
        CollectOutgoing(seq[i], &outgoing.dst_vars);
        pending_groups.clear();
        if (!outgoing.dst_vars.empty()) {
          pending_groups.push_back(std::move(outgoing));
        }
        has_any_aiu |= ContainsAIULoad(seq[i]);
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
      bool unknown_count = HasUnknownCount(pending_groups);
      for (int i = 0; i < static_cast<int>(seq.size()); ++i) {
        int newest_needed = FindNewestUsedGroup(seq[i], pending_groups);
        if (newest_needed >= 0) {
          int wait_count = (wait_all_ || unknown_count)
                               ? 0
                               : static_cast<int>(pending_groups.size()) -
                                     newest_needed - 1 + new_commits_before;
          if (needs_wait[i] < 0 || wait_count < needs_wait[i]) {
            needs_wait[i] = wait_count;
          }
          break;
        }
        if ((seq[i].as<ForNode>() || seq[i].as<SBlockRealizeNode>() ||
             seq[i].as<SBlockNode>()) &&
            ContainsAIULoad(seq[i])) {
          unknown_count = true;
        }
        // Count commit groups emitted before this point in the iteration.
        if (!wait_all_ && ContainsAIULoad(seq[i]) && !seq[i].as<ForNode>() &&
            !seq[i].as<SBlockRealizeNode>() && !seq[i].as<SBlockNode>() &&
            !IsAsyncScope(seq[i])) {
          new_commits_before += CountAIUGroups(seq[i]);
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
  const bool wait_all_;
  bool in_for_body_ = false;

  bool HasUnknownCount(const std::vector<PendingGroup> &groups) const {
    for (const auto &group : groups) {
      if (group.unknown_count) {
        return true;
      }
    }
    return false;
  }

  // May-pending destinations at a container exit. Only a proven wait-all
  // kills state; branch outputs are unioned. This summary deliberately does
  // not pretend a loop/branch batch is one runtime commit group.
  void CollectOutgoing(const Stmt &stmt,
                       std::unordered_set<const VarNode *> *pending) {
    if (IsWaitAll(stmt)) {
      pending->clear();
      return;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const auto &child : seq->seq) {
        CollectOutgoing(child, pending);
      }
      return;
    }
    if (const auto *branch = stmt.as<IfThenElseNode>()) {
      auto other = *pending;
      CollectOutgoing(branch->then_case, pending);
      if (branch->else_case.defined()) {
        CollectOutgoing(branch->else_case.value(), &other);
      }
      pending->insert(other.begin(), other.end());
      return;
    }
    if (const auto *loop = stmt.as<ForNode>()) {
      const int64_t *extent = as_const_int(loop->extent);
      if (extent && *extent <= 0) {
        return;
      }
      auto incoming = *pending;
      CollectOutgoing(loop->body, pending);
      // This kill-all/gen transfer reaches a fixed point after one iteration.
      // A dynamic loop can also execute zero times and retain incoming work.
      if (!extent) {
        pending->insert(incoming.begin(), incoming.end());
      }
      return;
    }
    if (const auto *block = stmt.as<SBlockRealizeNode>()) {
      auto skipped = *pending;
      CollectOutgoing(block->block, pending);
      if (!is_one(block->predicate)) {
        pending->insert(skipped.begin(), skipped.end());
      }
      return;
    }
    if (const auto *block = stmt.as<SBlockNode>()) {
      if (block->init.defined()) {
        auto skipped = *pending;
        CollectOutgoing(block->init.value(), pending);
        // Reduction initialization need not execute on every realization.
        pending->insert(skipped.begin(), skipped.end());
      }
      CollectOutgoing(block->body, pending);
      return;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (attr->attr_key == s_tir::attr::async_scope) {
        return;
      }
      if (attr->attr_key == s_tir::attr::async_wait_queue_scope &&
          is_zero(attr->value)) {
        const auto *count = attr->body.as<AttrStmtNode>();
        if (count &&
            count->attr_key == s_tir::attr::async_wait_inflight_count &&
            is_zero(count->value)) {
          pending->clear();
          CollectOutgoing(count->body, pending);
          return;
        }
      }
      if (attr->attr_key == s_tir::attr::async_commit_queue_scope) {
        CollectOutgoing(attr->body, pending);
        return;
      }
    }
    // Unknown scopes may issue work, but do not prove completion.
    std::vector<PendingGroup> groups;
    CollectAIUDstGroups(stmt, &groups);
    for (const auto &group : groups) {
      pending->insert(group.dst_vars.begin(), group.dst_vars.end());
    }
  }

  enum class IncomingState { kPending, kDrained, kConflict };

  bool IsWaitAll(const Stmt &stmt) const {
    const auto *eval = stmt.as<EvaluateNode>();
    const auto *call = eval ? eval->value.as<CallNode>() : nullptr;
    return call && call->op.same_as(builtin::ptx_wait_group()) &&
           call->args.size() == 1 && is_zero(call->args[0]);
  }

  // Track only groups committed before this statement, not new generations
  // issued inside it. Once a wait-all executes, those incoming groups are done.
  // Unrecognized constructs stay conservative; in particular pipeline-owned
  // scopes are opaque and wait<N> is not a proof of incoming-group completion.
  IncomingState
  CheckIncomingGroup(const Stmt &stmt,
                     const std::unordered_set<const VarNode *> &vars) const {
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &child : seq->seq) {
        auto state = CheckIncomingGroup(child, vars);
        if (state != IncomingState::kPending) {
          return state;
        }
      }
      return IncomingState::kPending;
    }
    if (const auto *branch = stmt.as<IfThenElseNode>()) {
      if (UsesAnyVar(branch->condition, vars)) {
        return IncomingState::kConflict;
      }
      auto then_state = CheckIncomingGroup(branch->then_case, vars);
      auto else_state =
          branch->else_case.defined()
              ? CheckIncomingGroup(branch->else_case.value(), vars)
              : IncomingState::kPending;
      if (then_state == IncomingState::kConflict ||
          else_state == IncomingState::kConflict) {
        return IncomingState::kConflict;
      }
      return then_state == IncomingState::kDrained &&
                     else_state == IncomingState::kDrained
                 ? IncomingState::kDrained
                 : IncomingState::kPending;
    }
    if (const auto *loop = stmt.as<ForNode>()) {
      if (UsesAnyVar(loop->min, vars) || UsesAnyVar(loop->extent, vars)) {
        return IncomingState::kConflict;
      }
      auto state = CheckIncomingGroup(loop->body, vars);
      const auto *extent = loop->extent.as<IntImmNode>();
      // A wait in a possibly zero-trip loop cannot protect a later statement.
      if (state == IncomingState::kDrained && !(extent && extent->value > 0)) {
        return IncomingState::kPending;
      }
      return state;
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      if (attr->attr_key == s_tir::attr::async_wait_queue_scope &&
          is_zero(attr->value)) {
        const auto *count = attr->body.as<AttrStmtNode>();
        if (count &&
            count->attr_key == s_tir::attr::async_wait_inflight_count &&
            is_zero(count->value)) {
          return IncomingState::kDrained;
        }
      }
      // These scopes execute their body unconditionally. Do not generalize
      // this to arbitrary attributes or pipeline-owned async_scope regions.
      if (attr->attr_key == s_tir::attr::async_commit_queue_scope) {
        return CheckIncomingGroup(attr->body, vars);
      }
    }
    if (IsWaitAll(stmt)) {
      return IncomingState::kDrained;
    }
    return UsesAnyVar(stmt, vars) ? IncomingState::kConflict
                                  : IncomingState::kPending;
  }

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

  int CountAIUGroups(const Stmt &stmt) {
    std::vector<PendingGroup> groups;
    CollectAIUDstGroups(stmt, &groups);
    return static_cast<int>(groups.size());
  }

  /*!
   * \brief Collect groups in program order, respecting commit scopes.
   * Mutually exclusive branches form one destination batch. In functions with
   * conditional AIU loads, wait<0> avoids relying on path-dependent group
   * counts. Destinations are extracted from tvm_access_ptr's buffer-data
   * argument.
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
      if (attr->attr_key == s_tir::attr::async_scope) {
        return;
      }
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
    for (const Call &call : CompilerAIULoadCollector().Collect(stmt)) {
      if (!call->args.empty()) {
        PendingGroup group;
        PrimExpr dst = call->args[0];
        // tvm_access_ptr(type, buffer_data, offset, extent, mask).
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
  }

  /*!
   * \brief Check if a statement uses any buffer whose data Var is in the set.
   * Checks BufferLoad, BufferStore, and tvm_access_ptr references.
   */
  bool UsesAnyVar(const ObjectRef &stmt,
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
    if (IsAsyncScope(stmt)) {
      return stmt;
    }
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
    auto *n = f.CopyOnWrite();
    // Conditional loads can issue different numbers of groups on each path.
    // Keep precise waits for straight-line loads; otherwise drain the queue at
    // every compiler-managed wait without counting commits across branches.
    bool wait_all = !CompilerAIULoadCollector(/*conditional_only=*/true)
                         .Collect(n->body)
                         .empty();
    n->body = AIUSyncBarrierInjector(wait_all)(n->body);
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
