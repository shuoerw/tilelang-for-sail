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
/*!
 * \brief AIU global-to-shared byte copy.
 *
 * ppu_aiu_load(smem_ptr, gmem_ptr, dim_c, dim_w, cube_c, cube_w,
 *              stride_w_bytes, start_c, start_w, swzl_mode,
 *              is_swizzle=1, prefetch_bytes=128)
 *
 * prefetch_bytes is a compile-time integer: 0, 64, 128, or 256.
 * Zero omits the LLC prefetch modifier.
 */
TVM_DLL const Op &ppu_aiu_load();

} // namespace tl
} // namespace tvm

#endif // TVM_TL_PPU_OP_BUILTIN_H_
