/*!
 * \file intrin_rule_ppu.cc
 * \brief PPU intrinsic rules.
 */
#include "support/check.h"
#include <tvm/ir/cast.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op_attr_types.h>

#include "target/intrin_rule.h"

namespace tvm {
namespace codegen {
namespace intrin {

using tirx::FLowerIntrinsic;
using namespace ffi;

struct PPUMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float()) {
      switch (t.bits()) {
      case 64:
        // Use nearbyint (ties-to-even) for round to match constant-folding
        // semantics.
        if (name == "round")
          return "nearbyint";
        return name;
      case 32:
        if (name == "round")
          return "nearbyintf";
        return name + 'f';
      case 16: {
        if (name == "fabs") {
          return "__habs";
        } else if (name == "round") {
          return "hrint";
        } else {
          return "h" + name;
        }
      }
      default:
        return "";
      }
    } else if (t.is_bfloat16()) {
      if (name == "fabs") {
        return "__habs";
      } else if (name == "round") {
        return "hrint";
      } else {
        return "h" + name;
      }
    } else if (t.is_int() || t.is_uint()) {
      switch (t.bits()) {
      case 32:
        return "__" + name;
      case 64:
        return "__" + name + "ll";
      default:
        return "";
      }
    }
    return "";
  }
};

struct PPUFastMath : public PPUMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float() && t.bits() == 32) {
      return "__" + name + 'f';
    } else {
      return PPUMath::operator()(t, name);
    }
    return "";
  }
};

struct PPUFastMathTan : public PPUMath {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_float()) {
      switch (t.bits()) {
      case 64:
        return name;
      // `__tanf` seems to produce some values too deviant from numpy tan
      // version. So, let's use just `tanf` instead.
      case 32:
        return name + 'f';
      case 16:
        return 'h' + name;
      default:
        return "";
      }
    }
    return "";
  }
};

struct PPUPopcount {
  std::string operator()(DataType t, std::string name) const {
    if (t.is_uint()) {
      switch (t.bits()) {
      case 32:
        return "__popc";
      case 64:
        return "__popcll";
      default:
        return "";
      }
    }
    return "";
  }
};

struct PPUWarpIntrinsic {
  const Op operator()(DataType t, const Op &orig_op) const {
    if (orig_op.same_as(builtin::tvm_warp_shuffle())) {
      return Op::Get("tirx.ppu.__shfl_sync");
    } else if (orig_op.same_as(builtin::tvm_warp_shuffle_up())) {
      return Op::Get("tirx.ppu.__shfl_up_sync");
    } else {
      ICHECK(orig_op.same_as(builtin::tvm_warp_shuffle_down()));
      return Op::Get("tirx.ppu.__shfl_down_sync");
    }
  }
};

static PrimExpr DispatchPPUWarpActiveMask(const PrimExpr &e) {
  const CallNode *call = e.as<CallNode>();
  return Call(call->dtype, Op::Get("tirx.ppu.__activemask"), call->args,
              call->annotations);
}

static PrimExpr DispatchPPUIsFinite(const PrimExpr &e) {
  const CallNode *call = e.as<CallNode>();
  ICHECK(call != nullptr);
  ICHECK_EQ(call->args.size(), 1U);

  DataType arg_dtype = call->args[0].dtype();
  if (arg_dtype.is_float() &&
      (arg_dtype.bits() == 32 || arg_dtype.bits() == 64)) {
    Array<PrimExpr> new_args = {StringImm("isfinite"), call->args[0]};
    return Call(call->dtype, builtin::call_pure_extern(), new_args,
                call->annotations);
  }

  return e;
}

template <typename T> static PrimExpr DispatchPPUShuffle(const PrimExpr &e) {
  const CallNode *call = e.as<CallNode>();
  ICHECK(call != nullptr);
  ICHECK_EQ(call->args.size(), 5); // mask, value, warp_id, width, warp_size
  Array<PrimExpr> ppu_args{
      {call->args[0], call->args[1], call->args[2], call->args[3]}};
  return Call(call->dtype, T()(call->dtype, Downcast<Op>(call->op)), ppu_args,
              call->annotations);
}

TVM_REGISTER_OP("tirx.tvm_warp_shuffle")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPPUShuffle<PPUWarpIntrinsic>);

TVM_REGISTER_OP("tirx.tvm_warp_shuffle_up")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPPUShuffle<PPUWarpIntrinsic>);

TVM_REGISTER_OP("tirx.tvm_warp_shuffle_down")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPPUShuffle<PPUWarpIntrinsic>);

TVM_REGISTER_OP("tirx.tvm_warp_activemask")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPPUWarpActiveMask);

TVM_REGISTER_OP("tirx.clz")
    .set_attr<FLowerIntrinsic>(
        "ppu.FLowerIntrinsic",
        DispatchPureExtern<PPUMath, /*dtype_from_arg=*/true>);

TVM_REGISTER_OP("tirx.floor")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.ceil")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.trunc")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.fabs")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.round")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.nearbyint")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.exp")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.rsqrt")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.exp2")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.exp10")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.erf")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.log")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.log2")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.log10")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.tan")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.cos")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.cosh")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.sin")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.sinh")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.atan")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.tanh")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.sqrt")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.pow")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.popcount")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUPopcount>);

TVM_REGISTER_OP("tirx.fmod")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic",
                               DispatchPureExtern<PPUMath>);

TVM_REGISTER_OP("tirx.isfinite")
    .set_attr<FLowerIntrinsic>("ppu.FLowerIntrinsic", DispatchPPUIsFinite);

// Register low-level builtin ops.
TVM_REGISTER_OP("tirx.ppu.__shfl_sync")
    .set_num_inputs(4)
    .add_argument("mask", "Expr", "The thread mask.")
    .add_argument("var", "Expr", "The variable to sync.")
    .add_argument("lane", "Expr", "The source thread id.")
    .add_argument("width", "Expr",
                  "The warp thread width, must be a power of 2.")
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "__shfl_sync")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<bool>("ppu.need_warp_shuffle", true);

TVM_REGISTER_OP("tirx.ppu.__shfl_up_sync")
    .set_num_inputs(4)
    .add_argument("mask", "Expr", "The thread mask.")
    .add_argument("var", "Expr", "The variable to sync.")
    .add_argument("delta", "Expr", "The source lane id offset to be added.")
    .add_argument("width", "Expr",
                  "The warp thread width, must be a power of 2.")
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "__shfl_up_sync")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<bool>("ppu.need_warp_shuffle", true);

TVM_REGISTER_OP("tirx.ppu.__shfl_down_sync")
    .set_num_inputs(4)
    .add_argument("mask", "Expr", "The thread mask.")
    .add_argument("var", "Expr", "The variable to sync.")
    .add_argument("delta", "Expr",
                  "The source lane id offset to be subtracted.")
    .add_argument("width", "Expr",
                  "The warp thread width, must be a power of 2.")
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "__shfl_down_sync")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<bool>("ppu.need_warp_shuffle", true);

TVM_REGISTER_OP("tirx.ppu.__activemask")
    .set_num_inputs(0)
    .set_attr<TGlobalSymbol>("TGlobalSymbol", "__activemask")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure))
    .set_attr<bool>("ppu.need_warp_shuffle", true);

} // namespace intrin
} // namespace codegen
} // namespace tvm
