/*!
 * \file tl/ppu/op/builtin.cc
 * \brief Registration of PPU-specific TileLang intrinsic Ops.
 */

#include "builtin.h"

#include <tvm/ir/transform.h>
#include <tvm/tirx/op_attr_types.h>

#include "op/builtin_registry.h"

namespace tvm {
namespace tl {

using namespace tirx;

TVM_REGISTER_PASS_CONFIG_OPTION(kDisableAIULower, Bool);
TVM_REGISTER_PASS_CONFIG_OPTION(kDisableLdmatSwzl, Bool);

TVM_REGISTER_OP("tl.clamp")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic", LowerClamp);

TIR_DEFINE_TL_BUILTIN(ppu_aiu_load)
    .set_num_inputs(-1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(tix_ldmatrix_b8)
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(tix_ldmatrix_swzl)
    .set_num_inputs(6)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TIR_DEFINE_TL_BUILTIN(ppu_to_uniform_b32)
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

} // namespace tl
} // namespace tvm

#undef TIR_DEFINE_TL_BUILTIN
