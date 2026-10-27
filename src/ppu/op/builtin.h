/*!
 * \file tl/ppu/op/builtin.h
 * \brief PPU-specific TileLang intrinsic Ops and pass configuration keys.
 */

#ifndef TVM_TL_PPU_OP_BUILTIN_H_
#define TVM_TL_PPU_OP_BUILTIN_H_

#include "cuda/op/builtin.h"

namespace tvm {
namespace tl {

static constexpr const char *kDisableAIULower = "tl.disable_aiu_lower";
static constexpr const char *kDisableLdmatSwzl = "tl.disable_ldmat_swzl";

TVM_DLL const Op &tix_ldmatrix_b8();
TVM_DLL const Op &tix_ldmatrix_swzl();
TVM_DLL const Op &ppu_to_uniform_b32();
TVM_DLL const Op &ppu_aiu_load();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_PPU_OP_BUILTIN_H_
