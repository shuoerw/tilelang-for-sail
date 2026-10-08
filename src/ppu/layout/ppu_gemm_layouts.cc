/*!
 * \file ppu/layout/ppu_gemm_layouts.cc
 * \brief PPU-specific shared-memory layout implementations for cutlass-ppu
 * Actlize layouts.
 *
 * These functions are migrated from src/layout/gemm_layouts.cc to isolate
 * PPU-specific logic from the common layout infrastructure.
 */

#include "ppu/layout/ppu_gemm_layouts.h"

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/tirx/op.h>

#include "layout/cute_layout.h"
#include "layout/layout.h"

#include "support/check.h"

namespace tvm {
namespace tl {

using namespace ffi;
using namespace tirx;

// ---------------------------------------------------------------------------
// Local helpers (mirrored from gemm_layouts.cc for self-contained compilation)
// ---------------------------------------------------------------------------

namespace {
struct SwizzleShapeInfo {
  int64_t stride;
  int64_t continuous;
  int element_size;
};

SwizzleShapeInfo GetSwizzleShapeInfoChecked(const Buffer &buffer) {
  ICHECK(buffer.defined()) << "Swizzle layout expects a defined buffer";
  ICHECK(buffer->shape.size() >= 2)
      << "Swizzle layout expects rank >= 2 buffer, got rank="
      << buffer->shape.size();
  size_t ndim = buffer->shape.size();
  auto stride = as_const_int(buffer->shape[ndim - 2]);
  auto continuous = as_const_int(buffer->shape[ndim - 1]);
  ICHECK(stride && continuous)
      << "Swizzle layout requires constant last-2 dims";
  return SwizzleShapeInfo{*stride, *continuous, buffer->dtype.bits()};
}

bool TryGetSwizzleShapeInfo(const Buffer &buffer, SwizzleShapeInfo *info) {
  if (!buffer.defined() || buffer->shape.size() < 2) {
    return false;
  }
  size_t ndim = buffer->shape.size();
  auto stride = as_const_int(buffer->shape[ndim - 2]);
  auto continuous = as_const_int(buffer->shape[ndim - 1]);
  if (!stride || !continuous) {
    return false;
  }
  *info = SwizzleShapeInfo{*stride, *continuous, buffer->dtype.bits()};
  return true;
}

} // namespace

static Layout ExpandLayout2D(const Layout &base, const Buffer &buffer) {
  Array<PrimExpr> leading_shape;
  leading_shape.reserve(buffer->shape.size() - 2);
  for (size_t i = 0; i + 2 < buffer->shape.size(); ++i) {
    leading_shape.push_back(buffer->shape[i]);
  }
  return base->Expand(leading_shape);
}

// ---------------------------------------------------------------------------
// XOR primitives (mirrored from gemm_layouts.cc)
// ---------------------------------------------------------------------------

static PrimExpr xor2x2(const PrimExpr &i, const PrimExpr &j) {
  return FloorMod(i + j, 2);
}

static PrimExpr xor4x4(const PrimExpr &i, const PrimExpr &j) {
  PrimExpr i0 = FloorMod(i, 2);
  PrimExpr j0 = FloorMod(j, 2);
  PrimExpr i1 = FloorDiv(i, 2);
  PrimExpr j1 = FloorDiv(j, 2);
  return 2 * xor2x2(i1, j1) + xor2x2(i0, j0);
}

static PrimExpr xor8x8(const PrimExpr &i, const PrimExpr j) {
  PrimExpr i0 = FloorMod(i, 2);
  PrimExpr j0 = FloorMod(j, 2);
  PrimExpr i1 = FloorDiv(i, 2);
  PrimExpr j1 = FloorDiv(j, 2);
  return 2 * xor4x4(i1, j1) + xor2x2(i0, j0);
}

// ---------------------------------------------------------------------------
// Common swizzle layouts (mirrored from gemm_layouts.cc — these are static
// in the original and therefore not linkable across translation units)
// ---------------------------------------------------------------------------

// Layout swizzling for 32 bytes
static Layout MakeQuarterBankSwizzleLayout2D(int stride, int continuous,
                                             int element_size) {
  Var i = InputPlaceholder(0);
  Var j = InputPlaceholder(1);
  int vector_size = 128 / element_size;
  ICHECK(stride == 4 || stride % 8 == 0) << "stride=" << stride;
  ICHECK(continuous % (vector_size * 2) == 0)
      << "continuous=" << continuous << ", vector_size=" << vector_size;
  PrimExpr ts = FloorDiv(i, 8);
  PrimExpr s = FloorMod(i, 8);
  PrimExpr tc = FloorDiv(FloorDiv(j, vector_size), 2);
  PrimExpr c = FloorMod(FloorDiv(j, vector_size), 2);
  PrimExpr vec = FloorMod(j, vector_size);
  PrimExpr c_swizzle = xor2x2(c, FloorDiv(s, 4));
  PrimExpr index = vec + (c_swizzle + s * 2) * vector_size;
  return Layout(Array<PrimExpr>{stride, continuous}, {tc, ts, index});
}

// Layout swizzling for 64 bytes
static Layout MakeHalfBankSwizzleLayout2D(int stride, int continuous,
                                          int element_size) {
  Var i = InputPlaceholder(0);
  Var j = InputPlaceholder(1);
  int vector_size = 128 / element_size;
  ICHECK(stride == 4 || stride % 8 == 0) << "stride=" << stride;
  ICHECK(continuous % (vector_size * 4) == 0)
      << "continuous=" << continuous << ", vector_size=" << vector_size;
  PrimExpr ts = FloorDiv(i, 8);
  PrimExpr s = FloorMod(i, 8);
  PrimExpr tc = FloorDiv(FloorDiv(j, vector_size), 4);
  PrimExpr c = FloorMod(FloorDiv(j, vector_size), 4);
  PrimExpr vec = FloorMod(j, vector_size);
  PrimExpr c_swizzle = xor4x4(c, FloorDiv(s, 2));
  PrimExpr index = vec + (c_swizzle + s * 4) * vector_size;
  return Layout(Array<PrimExpr>{stride, continuous}, {tc, ts, index});
}

// Layout swizzling for 128 bytes
static Layout MakeFullBankSwizzleLayout2D(int stride, int continuous,
                                          int element_size) {
  Var i = InputPlaceholder(0);
  Var j = InputPlaceholder(1);
  int vector_size = 128 / element_size;
  ICHECK(stride == 4 || stride % 8 == 0) << "stride=" << stride;
  ICHECK(continuous % (vector_size * 8) == 0)
      << "continuous=" << continuous << ", vector_size=" << vector_size;
  PrimExpr ts = FloorDiv(i, 8);
  PrimExpr s = FloorMod(i, 8);
  PrimExpr tc = FloorDiv(FloorDiv(j, vector_size), 8);
  PrimExpr c = FloorMod(FloorDiv(j, vector_size), 8);
  PrimExpr vec = FloorMod(j, vector_size);
  PrimExpr c_swizzle = xor8x8(c, s);
  PrimExpr index = vec + (c_swizzle + s * 8) * vector_size;
  return Layout(Array<PrimExpr>{stride, continuous}, {tc, ts, index});
}

static Layout MakeGemmABLayoutF64_Kinner(int stride, int continuous) {
  // Swizzle<2, 0, 4>
  Var i = InputPlaceholder(0);
  Var j = InputPlaceholder(1);
  PrimExpr tc = FloorDiv(j, 16);
  PrimExpr ts = FloorDiv(i, 4);
  PrimExpr c = FloorMod(j, 16);
  PrimExpr s = FloorMod(i, 4);
  PrimExpr swizzled_c = FloorDiv(c, 4) * 4 + xor4x4(FloorMod(c, 4), s);
  PrimExpr index = swizzled_c + s * 16;
  return Layout(Array<PrimExpr>{stride, continuous}, {tc, ts, index});
}

static Layout MakeGemmABLayoutF64_Kouter(int stride, int continuous) {
  // Swizzle<2, 2, 2>
  Var i = InputPlaceholder(0);
  Var j = InputPlaceholder(1);
  PrimExpr tc = FloorDiv(j, 16);
  PrimExpr ts = FloorDiv(i, 4);
  PrimExpr c = FloorMod(j, 16);
  PrimExpr s = FloorMod(i, 4);
  PrimExpr swizzled_c = FloorMod(c, 4) + xor4x4(FloorDiv(c, 4), s) * 4;
  PrimExpr index = swizzled_c + s * 16;
  return Layout(Array<PrimExpr>{stride, continuous}, {tc, ts, index});
}

// ---------------------------------------------------------------------------
// PPU-specific swizzle layouts
// ---------------------------------------------------------------------------

// PPU: layout swizzling for 64 bytes used by cutlass-ppu Actlize RS B layout.
static Layout MakeHalfBankSwizzleLayoutPPU2D(int stride, int continuous,
                                             int element_size, bool k_inner) {
  Var i = InputPlaceholder(0);
  Var j = InputPlaceholder(1);
  int vector_size = 128 / element_size;
  ICHECK(stride == 4 || stride % 8 == 0) << "stride=" << stride;
  ICHECK(continuous % (vector_size * 4) == 0)
      << "continuous=" << continuous << ", vector_size=" << vector_size;
  if (!k_inner) {
    PrimExpr ts = FloorDiv(i, 8);
    PrimExpr s_t = FloorMod(i, 8);
    PrimExpr s = FloorMod(s_t, 4) * 2 + FloorDiv(s_t, 4);
    PrimExpr tc = FloorDiv(FloorDiv(j, vector_size), 4);
    PrimExpr c = FloorMod(FloorDiv(j, vector_size), 4);
    PrimExpr vec = FloorMod(j, vector_size);
    PrimExpr c_swizzle = xor4x4(c, FloorDiv(s, 2));
    PrimExpr index = vec + (c_swizzle + s * 4) * vector_size;
    return Layout(Array<PrimExpr>{stride, continuous}, {tc, ts, index});
  }
  PrimExpr ts = FloorDiv(i, 8);
  PrimExpr s = FloorMod(i, 8);
  PrimExpr tc = FloorDiv(FloorDiv(j, vector_size), 4);
  PrimExpr c = FloorMod(FloorDiv(j, vector_size), 4);
  PrimExpr vec_t = FloorMod(j, vector_size);
  PrimExpr vec = FloorMod(vec_t, 4) * 2 + FloorDiv(vec_t, 4);
  PrimExpr c_swizzle = xor4x4(c, FloorDiv(s, 2));
  PrimExpr index = vec + (c_swizzle + s * 4) * vector_size;
  return Layout(Array<PrimExpr>{stride, continuous}, {tc, ts, index});
}

// PPU: layout swizzling for 128 bytes used by cutlass-ppu Actlize RS B layout.
static Layout MakeFullBankSwizzleLayoutPPU2D(int stride, int continuous,
                                             int element_size, bool k_inner) {
  Var i = InputPlaceholder(0);
  Var j = InputPlaceholder(1);
  int vector_size = 128 / element_size;
  ICHECK(stride == 4 || stride % 8 == 0) << "stride=" << stride;
  ICHECK(continuous % (vector_size * 8) == 0)
      << "continuous=" << continuous << ", vector_size=" << vector_size;
  if (!k_inner) {
    PrimExpr ts = FloorDiv(i, 8);
    PrimExpr s_t = FloorMod(i, 8);
    PrimExpr s = FloorMod(s_t, 4) * 2 + FloorDiv(s_t, 4);
    PrimExpr tc = FloorDiv(FloorDiv(j, vector_size), 8);
    PrimExpr c = FloorMod(FloorDiv(j, vector_size), 8);
    PrimExpr vec = FloorMod(j, vector_size);
    PrimExpr c_swizzle = xor8x8(c, s);
    PrimExpr index = vec + (c_swizzle + s * 8) * vector_size;
    return Layout(Array<PrimExpr>{stride, continuous}, {tc, ts, index});
  }
  PrimExpr ts = FloorDiv(i, 8);
  PrimExpr s = FloorMod(i, 8);
  PrimExpr tc = FloorDiv(FloorDiv(j, vector_size), 8);
  PrimExpr c = FloorMod(FloorDiv(j, vector_size), 8);
  PrimExpr vec_t = FloorMod(j, vector_size);
  PrimExpr vec = FloorMod(vec_t, 4) * 2 + FloorDiv(vec_t, 4);
  PrimExpr c_swizzle = xor8x8(c, s);
  PrimExpr index = vec + (c_swizzle + s * 8) * vector_size;
  return Layout(Array<PrimExpr>{stride, continuous}, {tc, ts, index});
}

// ---------------------------------------------------------------------------
// Public PPU layout functions
// ---------------------------------------------------------------------------

Layout MakeGemmBLayoutPaddedPPU(int stride, int continuous, int element_size,
                                bool k_inner) {
  // PPU: row/column swizzle used by cutlass-ppu Actlize shared B layouts.
  IterVar i = MakeIterVar("i", stride);
  IterVar j = MakeIterVar("j", continuous);
  int padded = continuous;
  if ((element_size * continuous) % 256 == 0)
    padded += 128 / element_size;
  if (!k_inner) {
    PrimExpr row_swizzled =
        FloorMod(i, 4) * 2 + FloorDiv(FloorMod(i, 8), 4) + FloorDiv(i, 8) * 8;
    return Layout(Array{i, j}, {row_swizzled * padded + j});
  }
  PrimExpr col_swizzled =
      FloorMod(j, 4) * 2 + FloorDiv(FloorMod(j, 8), 4) + FloorDiv(j, 8) * 8;
  return Layout(Array{i, j}, {i * padded + col_swizzled});
}

Layout MakeGemmABLayoutPPU(int mat_stride, int mat_continuous, int continuity,
                           int element_size, bool k_inner, bool is_gemm_rs) {
  // PPU: only the GEMM RS B-shared fp16 path needs the actlize swizzle.
  if (element_size == 64) {
    if (!k_inner && continuity % 16 == 0) // float64 KxN
      return MakeGemmABLayoutF64_Kouter(mat_stride, mat_continuous);
    if (k_inner && continuity % 16 == 0) // float64 NxK
      return MakeGemmABLayoutF64_Kinner(mat_stride, mat_continuous);
    return MakeGemmABLayoutPadded(mat_stride, mat_continuous, element_size);
  }
  int vector_size = 128 / element_size;
  if (!k_inner && element_size == 8) // int8 KxN
    return MakeGemmABLayoutPadded(mat_stride, mat_continuous, element_size);
  if (mat_continuous % (vector_size * 8) == 0) {
    if (is_gemm_rs && element_size == 16)
      return MakeFullBankSwizzleLayoutPPU2D(mat_stride, mat_continuous,
                                            element_size, k_inner);
    return MakeFullBankSwizzleLayout2D(mat_stride, mat_continuous,
                                       element_size);
  }
  if (mat_continuous % (vector_size * 4) == 0) {
    if (is_gemm_rs && element_size == 16)
      return MakeHalfBankSwizzleLayoutPPU2D(mat_stride, mat_continuous,
                                            element_size, k_inner);
    return MakeHalfBankSwizzleLayout2D(mat_stride, mat_continuous,
                                       element_size);
  }
  if (is_gemm_rs && element_size == 16)
    return MakeGemmBLayoutPaddedPPU(mat_stride, mat_continuous, element_size,
                                    k_inner);
  if (mat_continuous % (vector_size * 2) == 0)
    return MakeQuarterBankSwizzleLayout2D(mat_stride, mat_continuous,
                                          element_size);
  return MakeGemmABLayoutPadded(mat_stride, mat_continuous, element_size);
}

Layout MakePPUSwizzledLayout(const Buffer &buffer, bool k_inner, bool allow_pad,
                             bool is_gemm_rs) {
  // B-shared RS layout only when requested by the PPU GEMM implementation.
  auto info = GetSwizzleShapeInfoChecked(buffer);
  Layout base;
  if (allow_pad) {
    base = MakeGemmABLayoutPPU(static_cast<int>(info.stride),
                               static_cast<int>(info.continuous),
                               static_cast<int>(info.continuous),
                               info.element_size, k_inner, is_gemm_rs);
  } else {
    base = MakeGemmABLayoutHopper(
        static_cast<int>(info.stride), static_cast<int>(info.continuous),
        static_cast<int>(info.continuous), info.element_size, k_inner);
  }
  return ExpandLayout2D(base, buffer);
}

AIUSharedLayoutClass ClassifySharedLayoutForAIU(const Layout &layout,
                                                const Buffer &buffer) {
  const int bits = static_cast<int>(buffer->dtype.bits());
  if (bits <= 0 || (bits & (bits - 1)) != 0) {
    return AIUSharedLayoutClass::kOther;
  }
  auto can_recast_bytes = [bits](const cute::ComposedLayout &value) {
    if (bits >= 8)
      return true;
    const int factor = 8 / bits;
    if (value->offset % factor != 0)
      return false;
    // An AIU byte copy cannot implement nibble permutations or row strides
    // that cut through a byte. Recast rounds those strides up.
    if (value->swizzle->IsSwizzled() &&
        (int64_t(1) << value->swizzle->m_base) < factor)
      return false;
    auto shape = cute::Flatten(value->layout->shape);
    auto stride = cute::Flatten(value->layout->stride);
    for (int i = 0; i < cute::Rank(shape); ++i) {
      if (!cute::IsConst(shape[i]) || !cute::IsConst(stride[i]))
        return false;
      const int64_t size = cute::AsConst(shape[i]);
      const int64_t step = cute::AsConst(stride[i]);
      if (size <= 1 || step == 0)
        continue;
      if (step == 1) {
        if (size % factor != 0)
          return false;
      } else if (step % factor != 0) {
        return false;
      }
    }
    return true;
  };
  auto composed = cute::ComposedLayoutFromTileLang(layout);
  if (!composed.defined() || !can_recast_bytes(composed.value())) {
    return AIUSharedLayoutClass::kOther;
  }
  auto composed_bytes =
      composed.value().Recast(static_cast<int>(buffer->dtype.bits()), 8);
  const auto *sw = composed_bytes->swizzle.get();
  if (!sw->IsSwizzled()) {
    Layout linear_ref = MakeLinearLayout(buffer->shape);
    auto ref = cute::ComposedLayoutFromTileLang(linear_ref);
    if (!ref.defined() || !can_recast_bytes(ref.value())) {
      return AIUSharedLayoutClass::kOther;
    }
    auto ref_bytes =
        ref.value().Recast(static_cast<int>(buffer->dtype.bits()), 8);
    if (!ref_bytes->swizzle->IsSwizzled() &&
        composed_bytes->offset == ref_bytes->offset &&
        StructuralEqual()(composed_bytes->layout, ref_bytes->layout)) {
      return AIUSharedLayoutClass::kLinear;
    }
    return AIUSharedLayoutClass::kOther;
  }
  if (sw->m_base != 4 || sw->s_shift != 3 || sw->b_bits < 1 || sw->b_bits > 3) {
    return AIUSharedLayoutClass::kOther;
  }
  SwizzleShapeInfo info;
  if (!TryGetSwizzleShapeInfo(buffer, &info) ||
      (info.stride != 4 && info.stride % 8 != 0) ||
      info.continuous * info.element_size % (128 << sw->b_bits) != 0) {
    return AIUSharedLayoutClass::kOther;
  }
  Layout canonical = sw->b_bits == 1   ? MakeQuarterBankSwizzleLayout(buffer)
                     : sw->b_bits == 2 ? MakeHalfBankSwizzleLayout(buffer)
                                       : MakeFullBankSwizzleLayout(buffer);
  auto ref = cute::ComposedLayoutFromTileLang(canonical);
  if (!ref.defined() || !can_recast_bytes(ref.value())) {
    return AIUSharedLayoutClass::kOther;
  }
  auto ref_bytes =
      ref.value().Recast(static_cast<int>(buffer->dtype.bits()), 8);
  // AIU fixes both the XOR pattern and the arrangement of rows and column
  // blocks. Matching the XOR pattern alone does not establish compatibility.
  if (composed_bytes->offset != ref_bytes->offset ||
      !StructuralEqual()(composed_bytes->layout, ref_bytes->layout)) {
    return AIUSharedLayoutClass::kOther;
  }
  return sw->b_bits == 1   ? AIUSharedLayoutClass::kSwizzle32
         : sw->b_bits == 2 ? AIUSharedLayoutClass::kSwizzle64
                           : AIUSharedLayoutClass::kSwizzle128;
}

// ---------------------------------------------------------------------------
// FFI registration
// ---------------------------------------------------------------------------

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def(
      "tl.make_ppu_swizzled_layout",
      [](const Buffer &buffer, bool k_inner, bool allow_pad, bool is_gemm_rs) {
        return MakePPUSwizzledLayout(buffer, k_inner, allow_pad, is_gemm_rs);
      });
}

} // namespace tl
} // namespace tvm
