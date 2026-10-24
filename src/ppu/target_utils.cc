/*!
 * \file tl/ppu/target_utils.cc
 * \brief PPU target attribute helpers.
 */

#include "ppu/target_utils.h"

#include <tvm/ffi/reflection/registry.h>

#include <string>

#include "dlpack/dlpack.h"
#include "support/check.h"

namespace tvm {
namespace tl {

int GetPPUArchInt(Target target) {
  if (!TargetIsPPU(target))
    return 0;
  auto s = target->GetAttr<ffi::String>("arch");
  ICHECK(s.has_value());
  const std::string arch_str = s.value();
  size_t pos = arch_str.rfind('_');
  ICHECK(pos != std::string::npos && pos + 1 < arch_str.size())
      << "arch string must contain '_' followed by a version number";
  const std::string suffix = arch_str.substr(pos + 1);
  ICHECK(!suffix.empty() &&
         suffix.find_first_not_of("0123456789") == std::string::npos)
      << "arch version suffix must be a number";
  return std::stoi(suffix);
}

bool TargetIsPPU(Target target) {
  return target->GetTargetDeviceType() == kDLPPU;
}

bool TargetHasAiuCopy(Target target) {
  if (!TargetIsPPU(target))
    return false;
  int arch = GetPPUArchInt(target);
  return arch >= 15;
}

bool TargetPPUHasAsyncCopy(Target target) {
  if (!TargetIsPPU(target))
    return false;
  return true;
}

int TargetPPUGetWarpSize(Target target) {
  (void)target;
  return 32;
}

bool TargetPPUHasLdmatrix(Target target) {
  if (!TargetIsPPU(target))
    return false;
  return true;
}

bool TargetPPUHasLdmatrixSwzl(const Target &target) {
  return TargetIsPPU(target) && GetPPUArchInt(target) == 15;
}

bool TargetPPUHasStmatrix(Target target) { return false; }

bool IsPpuVectorizableFP8(DataType dtype) {
  // NOTE: E8M0 is a special type of FP8 which is not handled here.
  // We only handle FP8 types which can be represented with
  // the PPU fp8 conversion intrinsics here.
  return dtype.is_float8_e4m3() || dtype.is_float8_e4m3fn() ||
         dtype.is_float8_e5m2();
}

bool IsPpuVectorizableCast(DataType from_ty, DataType target_ty) {
  // float16 -> float32
  if (from_ty.is_float16() && target_ty.is_float() && target_ty.bits() == 32)
    return true;

  // float32 -> float16
  if (from_ty.is_float() && from_ty.bits() == 32 && target_ty.is_float16())
    return true;

  // bfloat16 -> float32
  if (from_ty.is_bfloat16() && target_ty.is_float() && target_ty.bits() == 32)
    return true;

  // float32 -> bfloat16
  if (from_ty.is_float() && from_ty.bits() == 32 && target_ty.is_bfloat16())
    return true;

  // float32 -> float8 (E4M3/E5M2)
  if (from_ty.is_float() && from_ty.bits() == 32 &&
      IsPpuVectorizableFP8(target_ty))
    return true;

  // float8 (E4M3/E5M2) -> float32
  if (IsPpuVectorizableFP8(from_ty) && target_ty.is_float() &&
      target_ty.bits() == 32)
    return true;

  // float8 (E4M3/E5M2) -> float16
  if (IsPpuVectorizableFP8(from_ty) && target_ty.is_float16())
    return true;

  // float8 (E4M3/E5M2) -> bfloat16
  if (IsPpuVectorizableFP8(from_ty) && target_ty.is_bfloat16())
    return true;

  // float16 -> float8 (E4M3/E5M2)
  if (from_ty.is_float16() && IsPpuVectorizableFP8(target_ty))
    return true;

  // bfloat16 -> float8 (E4M3/E5M2)
  if (from_ty.is_bfloat16() && IsPpuVectorizableFP8(target_ty))
    return true;

  return false;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.TargetIsPPU", [](Target target) { return TargetIsPPU(target); })
      .def("tl.TargetHasAiuCopy",
           [](Target target) { return TargetHasAiuCopy(target); })
      .def("tl.TargetPPUGetWarpSize",
           [](Target target) { return TargetPPUGetWarpSize(target); })
      .def("tl.TargetPPUHasAsyncCopy",
           [](Target target) { return TargetPPUHasAsyncCopy(target); })
      .def("tl.TargetPPUHasLdmatrix",
           [](Target target) { return TargetPPUHasLdmatrix(target); })
      .def("tl.TargetPPUHasStmatrix",
           [](Target target) { return TargetPPUHasStmatrix(target); });
}

} // namespace tl
} // namespace tvm
