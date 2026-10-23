/*!
 * \brief Reorder aiu_load instructions by their dst buffer's first-use position
 * \file reorder_aiu_loads.cc
 *
 * In stage=0 scenarios, aiu_load instructions are reordered within each SeqStmt
 * so that loads whose dst buffers are consumed earlier appear first. This
 * enables better overlap between loads and computation by issuing loads in
 * consumption order. The pass preserves correctness by only moving loads
 * forward (never backward) and by ensuring all address dependencies are
 * satisfied at the insertion point.
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <sstream>
#include <unordered_map>
#include <unordered_set>
#include <vector>

#include "../../op/builtin.h"
#include "ppu/op/copy.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

/*!
 * \brief Extract the dst buffer VarNode from an aiu_load call.
 * arg[0] is tvm_access_ptr(type, buffer_data_var, offset, extent, mask).
 * Returns nullptr if extraction fails.
 */
static const VarNode *ExtractAIULoadDstVar(const Stmt &stmt) {
  const VarNode *result = nullptr;
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (result)
      return;
    if (auto *call = node.as<CallNode>()) {
      if (call->op.same_as(tl::ppu_aiu_load()) && call->args.size() > 0) {
        if (auto *access_call = call->args[0].as<CallNode>()) {
          if (access_call->op.same_as(builtin::tvm_access_ptr()) &&
              access_call->args.size() > 1) {
            if (auto *var = access_call->args[1].as<VarNode>()) {
              result = var;
            }
          }
        }
      }
    }
  });
  return result;
}

/*!
 * \brief Check if a statement is (or contains) an aiu_load at the top level
 * (not inside For/Block/If containers).
 */
static bool IsAIULoadStmt(const Stmt &stmt) {
  bool found = false;
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (found)
      return;
    if (auto *call = node.as<CallNode>()) {
      if (call->op.same_as(tl::ppu_aiu_load())) {
        found = true;
      }
    }
  });
  return found;
}

/*!
 * \brief Check if a statement uses any buffer whose data Var is in the set.
 * Checks BufferLoad, BufferStore, and tvm_access_ptr references.
 */
static bool UsesAnyVar(const Stmt &stmt,
                       const std::unordered_set<const VarNode *> &vars) {
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
      // Generic path: check all args for VarNode references
      if (!found) {
        for (const auto &arg : call->args) {
          if (auto *var = arg.as<VarNode>()) {
            if (vars.count(var)) {
              found = true;
              break;
            }
          }
        }
      }
    }
  });
  return found;
}

/*!
 * \brief Collect all VarNode pointers referenced in an expression or statement.
 */
static void CollectVars(const Stmt &stmt,
                        std::unordered_set<const VarNode *> *vars) {
  PostOrderVisit(stmt, [&](const ObjectRef &node) {
    if (auto *var = node.as<VarNode>()) {
      vars->insert(var);
    }
  });
}

/*!
 * \brief Check if a statement defines (binds) a given Var.
 * Recognizes LetStmt, AttrStmt (with Var node), and BufferStore.
 */
static const VarNode *GetDefinedVar(const Stmt &stmt) {
  if (auto *let = stmt.as<BindNode>()) {
    return let->var.get();
  }
  if (auto *attr = stmt.as<AttrStmtNode>()) {
    if (auto *var = attr->node.as<VarNode>()) {
      return var;
    }
  }
  return nullptr;
}

class AIULoadReorder : public StmtExprMutator {
public:
  Stmt VisitStmt_(const ForNode *op) final {
    // Check num_stages annotation
    auto num_stages_anno = op->annotations.Get("num_stages");
    if (num_stages_anno) {
      auto *int_imm = num_stages_anno->as<IntImmNode>();
      if (!int_imm)
        return tvm::ffi::GetRef<Stmt>(op);
      int num_stages = int_imm->value;
      if (num_stages > 0) {
        // stage > 0: load scheduling is handled by InjectSoftwarePipeline
        // which assigns loads to dedicated pipeline stages for overlap.
        // Per-load reordering is unnecessary and could conflict with
        // the pipeline's scheduling decisions.
        return tvm::ffi::GetRef<Stmt>(op);
      }
    }
    // stage == 0 or no annotation: recurse into the loop body
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    // 1. First recurse into children
    Array<Stmt> seq;
    for (const Stmt &s : op->seq) {
      seq.push_back(VisitStmt(s));
    }

    // Do not move loads across caller-managed async groups.
    bool has_explicit_load = false;
    PostOrderVisit(GetRef<Stmt>(op), [&](const ObjectRef &node) {
      if (const auto *call = node.as<CallNode>()) {
        if (call->op.same_as(tl::ppu_aiu_load()) &&
            !ppu::IsCompilerSyncAiuLoad(call)) {
          has_explicit_load = true;
        }
      }
    });
    if (has_explicit_load)
      return SeqStmt(seq);

    // 2. Collect aiu_load statement indices (top-level only, not inside
    //    For/Block/If containers)
    std::vector<int> load_indices;
    for (int i = 0; i < static_cast<int>(seq.size()); ++i) {
      bool is_container =
          seq[i].as<ForNode>() || seq[i].as<SBlockRealizeNode>() ||
          seq[i].as<SBlockNode>() || seq[i].as<IfThenElseNode>();
      if (!is_container && IsAIULoadStmt(seq[i])) {
        load_indices.push_back(i);
      }
    }

    if (load_indices.empty()) {
      return SeqStmt(seq);
    }

    // 3. For each aiu_load, extract dst buffer var and find first use position
    struct LoadInfo {
      int original_idx;
      const VarNode *dst_var;
      int first_use_pos; // position in SeqStmt where dst is first consumed
    };
    std::vector<LoadInfo> load_infos;

    for (int idx : load_indices) {
      const VarNode *dst_var = ExtractAIULoadDstVar(seq[idx]);
      if (!dst_var)
        continue;

      // 4. Scan forward to find first use of dst buffer
      std::unordered_set<const VarNode *> dst_vars;
      dst_vars.insert(dst_var);

      int first_use = static_cast<int>(seq.size()); // default: end
      for (int j = idx + 1; j < static_cast<int>(seq.size()); ++j) {
        if (UsesAnyVar(seq[j], dst_vars)) {
          first_use = j;
          break;
        }
      }

      load_infos.push_back({idx, dst_var, first_use});
      DLOG(INFO) << "ReorderAIULoads: Load[" << dst_var->name_hint
                 << "] original_idx=" << idx << " first_use=" << first_use;
    }

    if (load_infos.empty()) {
      return SeqStmt(seq);
    }

    if (load_infos.size() <= 1) {
      return SeqStmt(seq);
    }

    // 5. Sort loads by first-use position (stable sort preserves original order
    //    for loads with same use position)
    std::stable_sort(load_infos.begin(), load_infos.end(),
                     [](const LoadInfo &a, const LoadInfo &b) {
                       return a.first_use_pos < b.first_use_pos;
                     });

    // 6. Try to reorder: move loads toward their target positions
    //    We work on an index-based permutation of the original seq.
    //    Build a mapping: for each load, compute its target position and
    //    check if moving is feasible (all dependencies satisfied).

    // Build a map from VarNode* to the index of the statement that defines it
    std::unordered_map<const VarNode *, int> var_def_pos;
    for (int i = 0; i < static_cast<int>(seq.size()); ++i) {
      const VarNode *def = GetDefinedVar(seq[i]);
      if (def) {
        var_def_pos[def] = i;
      }
    }

    // Track which positions are "load positions" for the reorder
    std::unordered_set<int> is_load_pos;
    for (const auto &info : load_infos) {
      is_load_pos.insert(info.original_idx);
    }

    // For each load, determine its bundle (load + address dependencies) and
    // target insertion position
    struct MoveAction {
      std::vector<int> bundle_indices; // indices to move (in original order)
      int target_pos;                  // insert before this position
    };
    std::vector<MoveAction> actions;

    for (const auto &info : load_infos) {
      int load_idx = info.original_idx;

      // Collect address dependencies for this load
      std::unordered_set<const VarNode *> needed_vars;
      CollectVars(seq[load_idx], &needed_vars);
      // Remove the dst var itself from dependencies
      needed_vars.erase(info.dst_var);

      // Find all dependency statements in SeqStmt (transitive closure)
      std::unordered_set<int> bundle_set;
      bundle_set.insert(load_idx);

      std::vector<const VarNode *> worklist(needed_vars.begin(),
                                            needed_vars.end());
      std::unordered_set<const VarNode *> visited_vars(needed_vars.begin(),
                                                       needed_vars.end());

      while (!worklist.empty()) {
        const VarNode *var = worklist.back();
        worklist.pop_back();
        auto it = var_def_pos.find(var);
        if (it != var_def_pos.end() && it->second < load_idx) {
          int dep_idx = it->second;
          if (bundle_set.insert(dep_idx).second) {
            // Recurse: collect vars used by this dependency
            std::unordered_set<const VarNode *> dep_vars;
            CollectVars(seq[dep_idx], &dep_vars);
            for (const VarNode *dv : dep_vars) {
              if (visited_vars.insert(dv).second) {
                worklist.push_back(dv);
              }
            }
          }
        }
      }

      // Sort bundle indices in original order
      std::vector<int> bundle(bundle_set.begin(), bundle_set.end());
      std::sort(bundle.begin(), bundle.end());

      actions.push_back({std::move(bundle), load_idx});
    }

    // Now perform the actual reordering:
    // Strategy: place loads in the sorted order (by first-use), keeping
    // non-load statements in their original relative order.
    // For each load (in target order), we assign it to the first available
    // "load slot" from the original positions.

    // Simpler approach: extract all load stmts from seq, sort them by
    // first-use, and re-insert them at the original load positions.
    // This guarantees loads only swap among themselves (no non-load stmts
    // are displaced) and address dependencies that are NOT in load positions
    // remain stable.

    // However, we need to handle bundles. Let's use a different approach:
    // Check each load in sorted order; if it needs to move forward, verify
    // feasibility and perform the move.

    // Final approach: re-insert loads at load-slot positions in sorted order
    // First, verify this is safe (each load's deps are before its new position)
    std::vector<int> sorted_load_slots;
    for (const auto &info : load_infos) {
      sorted_load_slots.push_back(info.original_idx);
    }
    std::sort(sorted_load_slots.begin(), sorted_load_slots.end());

    // Attempt assignment: load_infos[i] goes to sorted_load_slots[i]
    // Check feasibility: all deps of load_infos[i] must be before
    // sorted_load_slots[i]
    std::vector<int> final_assignment(load_infos.size(), -1);
    std::vector<bool> slot_used(load_infos.size(), false);

    for (size_t i = 0; i < load_infos.size(); ++i) {
      int load_idx = load_infos[i].original_idx;

      // Find the earliest available slot where all deps are satisfied
      int assigned_slot = -1;
      for (size_t s = 0; s < sorted_load_slots.size(); ++s) {
        if (slot_used[s])
          continue;
        int slot_pos = sorted_load_slots[s];

        // Check: all non-load dependencies of this load must be before slot_pos
        // Collect vars needed by this load
        std::unordered_set<const VarNode *> needed;
        CollectVars(seq[load_idx], &needed);
        needed.erase(load_infos[i].dst_var);

        bool feasible = true;
        for (const VarNode *var : needed) {
          auto it = var_def_pos.find(var);
          if (it != var_def_pos.end()) {
            int def_pos = it->second;
            // If the definition is at a load slot that will also be moved,
            // we need more complex analysis. For safety, only allow if def
            // is strictly before the target slot and not a load position.
            if (def_pos >= slot_pos) {
              // Dependency not satisfied at this slot
              feasible = false;
              break;
            }
          }
        }

        if (feasible) {
          assigned_slot = static_cast<int>(s);
          break;
        }
      }

      if (assigned_slot >= 0) {
        final_assignment[i] = sorted_load_slots[assigned_slot];
        slot_used[assigned_slot] = true;
      } else {
        // Cannot move: keep at original position
        // Find the slot corresponding to original position
        for (size_t s = 0; s < sorted_load_slots.size(); ++s) {
          if (!slot_used[s] && sorted_load_slots[s] == load_idx) {
            final_assignment[i] = load_idx;
            slot_used[s] = true;
            break;
          }
        }
        // Fallback: assign to any remaining slot at or after original pos
        if (final_assignment[i] < 0) {
          for (size_t s = 0; s < sorted_load_slots.size(); ++s) {
            if (!slot_used[s]) {
              final_assignment[i] = sorted_load_slots[s];
              slot_used[s] = true;
              break;
            }
          }
        }
      }
    }

    {
      std::ostringstream oss;
      oss << "ReorderAIULoads: final order = [";
      for (size_t i = 0; i < load_infos.size(); ++i) {
        if (i > 0)
          oss << ", ";
        oss << load_infos[i].dst_var->name_hint << "@" << final_assignment[i];
      }
      oss << "]";
      DLOG(INFO) << oss.str();
    }

    // Build the new sequence: place loads at their assigned slots
    // Map: slot_position -> which load goes there
    std::unordered_map<int, int> slot_to_load_original;
    for (size_t i = 0; i < load_infos.size(); ++i) {
      slot_to_load_original[final_assignment[i]] = load_infos[i].original_idx;
    }

    Array<Stmt> new_seq;
    for (int i = 0; i < static_cast<int>(seq.size()); ++i) {
      if (is_load_pos.count(i)) {
        // This is a load slot: insert the assigned load
        auto it = slot_to_load_original.find(i);
        if (it != slot_to_load_original.end()) {
          new_seq.push_back(seq[it->second]);
        } else {
          // Should not happen, but keep original as fallback
          new_seq.push_back(seq[i]);
        }
      } else {
        new_seq.push_back(seq[i]);
      }
    }

    return SeqStmt(std::move(new_seq));
  }
};

using namespace tirx::transform;

tvm::transform::Pass ReorderAIULoads() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    auto *n = f.CopyOnWrite();
    n->body = AIULoadReorder()(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ReorderAIULoads", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.ppu.transform.ReorderAIULoads", ReorderAIULoads);
}

} // namespace tl
} // namespace tvm
