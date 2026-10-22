/*!
 * \file target/codegen.h
 * \brief Utility to generate code
 */
#ifndef TVM_TL_TARGET_CODEGEN_PPU_H_
#define TVM_TL_TARGET_CODEGEN_PPU_H_

#include "support/check.h"
#include <optional>
#include <tvm/target/codegen.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>

#include <string>
#include <unordered_map>
#include <unordered_set>

#include "target/source/codegen_c.h"

namespace tvm {
namespace codegen {

namespace ppu {

class CodeGenTileLangPPU final : public CodeGenC {
public:
  CodeGenTileLangPPU();
  std::string Finish();
  // override behavior
  void PrintFuncPrefix(std::ostream &os) final;
  void PrintExtraAttrs(const PrimFunc &f);
  void VisitStmt_(const ForNode *op) final;
  void PrintStorageSync(const CallNode *op) final;
  void PrintStorageScope(const std::string &scope,
                         std::ostream &os) final; // NOLINT(*)
  void PrintVecBinaryOp(const std::string &op, DataType t, PrimExpr lhs,
                        PrimExpr rhs,
                        std::ostream &os) final;      // NOLINT(*)
  void PrintType(DataType t, std::ostream &os) final; // NOLINT(*)
  void PrintVecElemLoad(const std::string &vec, DataType t, int i,
                        std::ostream &os) final; // NOLINT(*)
  void PrintVecElemStore(const std::string &vec, DataType t, int i,
                         const std::string &value) final;
  std::string GetVecLoad(DataType t, const BufferNode *buffer,
                         PrimExpr base) final;
  void PrintVecStore(const BufferNode *buffer, DataType t, PrimExpr base,
                     const std::string &value) final;
  void BindThreadIndex(const IterVar &iv) final; // NOLINT(*)
  void PrintVecElemLoadExpr(DataType t, int i, const std::string &value,
                            std::ostream &os) final;
  std::string CastFromTo(std::string value, DataType from,
                         DataType target) final;
  // overload visitor
  void VisitExpr_(const RampNode *op, std::ostream &os) final;      // NOLINT(*)
  void VisitExpr_(const BroadcastNode *op, std::ostream &os) final; // NOLINT(*)
  void VisitExpr_(const FloatImmNode *op, std::ostream &os) final;
  void VisitExpr_(const CallNode *op, std::ostream &os) final;
  void VisitExpr_(const CastNode *op, std::ostream &os) final;
  void VisitExpr_(const ShuffleNode *op, std::ostream &os) final;
  void VisitExpr_(const MinNode *op, std::ostream &os) final;
  void VisitExpr_(const MaxNode *op, std::ostream &os) final;
  void VisitStmt_(const EvaluateNode *op) final;
  void VisitStmt_(const AllocBufferNode *op) final;
  void VisitStmt_(const AttrStmtNode *op) final;
  void VisitExpr_(const BufferLoadNode *op, std::ostream &os) final;
  void VisitStmt_(const BufferStoreNode *op) final;

  // Override this as a work around for __grid_constant__ parameter
  void AddFunction(const GlobalVar &gvar, const PrimFunc &f);
  void PrintFunctionSignature(const ffi::String &function_name,
                              const PrimFunc &func, std::ostream &os);

protected:
  void ReserveKeywordsAsUnique_();
  virtual std::string GetBufferRef(DataType t, const BufferNode *buffer,
                                   PrimExpr index) final;
  void PrintCallExtern(Type ret_type, ffi::String global_symbol,
                       const ffi::Array<PrimExpr> &args, bool skip_first_arg,
                       std::ostream &os) final; // NOLINT(*)

private:
  // Handle volatile loads
  void HandleVolatileLoads(const std::string &value, const BufferLoadNode *op,
                           std::ostream &os) final;
  bool HandleLateIntrinsicCall(const CallNode *op, std::ostream &os);

  // Whether scope such as "__shared__" or "__constant__"  is part of type.
  bool IsScopePartOfType() const final { return false; }

  friend void PrintConst(const FloatImmNode *op, std::ostream &os,
                         CodeGenTileLangPPU *p);

  // Global acrand state
  std::string acrand_random_generator_state;
  std::string acrand_random_generator_state_type;
  // Map from tl.rng_init call to the function-scope RNG state variable name.
  std::unordered_map<const CallNode *, std::string> rng_state_name_map_;

  // whether enable fp16
  bool enable_fp16_{false};
  // whether enable bf16
  bool enable_bf16_{false};
  // whether enable fp8
  bool enable_fp8_{false};
  // whether enable fp6
  bool enable_fp6_{false};
  // whether enable fp4
  bool enable_fp4_{false};
  // whether enable int8
  bool enable_int8_{false};
  // whether enable warp shuffle intrinsics
  bool enable_warp_shuffle_{false};
  // whether need math_constants.h
  bool need_math_constants_h_{false};
  // whether need tl math header (hexp/hlog/hsin/hcos fast-math macros)
  bool need_math_h_{false};
  // whether need mma.h
  bool need_mma_h_{false};
  // whether need tl mma instruction header
  bool need_mma_instruction_h_{false};
  // whether need tl mma_sp instruction header
  bool need_mma_sp_instruction_h_{false};
  // whether need tl intrin header (get_lane_idx/get_warp_idx*/tl_shuffle_elect)
  bool need_intrin_h_{false};
  // whether need tl barrier header (mbarrier_cp_async_arrive*)
  bool need_barrier_h_{false};
  // whether need cast_smem_ptr_to_int helper function
  bool need_cast_smem_ptr_to_int_{false};
  // PPU: whether the current function uses Actlize MMA (m16n16k*)
  // instructions, which require the PPU-specific ldmatrix layout.
  bool use_actlize_ldmatrix_{false};
  // whether need cooperative_groups.h
  bool need_cooperative_groups_{false};
  // whether need acrand_kernel.h
  bool need_acrand_kernel_h_{false};
  // Op attribute map
  OpAttrMap<bool> op_need_warp_shuffle_ =
      Op::GetAttrMap<bool>("ppu.need_warp_shuffle");

  // The name of the barrier array in shared memory
  const std::string barrier_name_ = "barrier";
  // The size of the barrier array in shared memory
  int barrier_count_ = -1;
  // The name of the mbarrier array in shared memory
  // The same as injected_mbarrier_name_ in transform/common/mbarrier.h
  const std::string mbarrier_name_ = "mbarrier";
  // The type name of the mbarrier array
  const std::string mbarrier_dtype_ = "Barrier";
  // The alignment of the barrier array in shared memory
  // Set to 16 to maintain minimum alignment requirements for async bulk copy
  const int barrier_alignment_bytes_ = 16;

  std::unordered_map<const VarNode *, std::string> fragment_shapes;
  std::unordered_map<const VarNode *, std::string> fragment_layouts;
  std::unordered_map<const VarNode *, IntImm> unroll_factor;
  std::optional<std::tuple<int64_t, int64_t, int64_t>> cluster_dims;
  // ffi::Map from VarNode to packed buffer variable name for fp4 packed storage
  std::unordered_map<const VarNode *, std::string> fp4_packed_buffers_;
  friend void PrintConst(const FloatImmNode *op, std::ostream &os,
                         CodeGenTileLangPPU *p);
  void PrintWmmaScope(const std::string &scope, DataType t,
                      const VarNode *variable, std::ostream &os);
  int32_t GetWmmaFragmentSize(const std::string &scope, const VarNode *variable,
                              int32_t size);

  std::vector<std::string> eviction_policy_names_ = {
      "EVICT_NORMAL", "EVICT_FIRST", "EVICT_LAST"};
  std::unordered_set<std::string> bf16_supported_ops_ = {
      "bf1622float2", "bf1622int16", "float22bf162", "bf162bf162"};
};

} // namespace ppu
} // namespace codegen
} // namespace tvm

#endif // TVM_TL_TARGET_CODEGEN_PPU_H_
