/*!
 * \file ppu/layout/ppu_gemm_layouts.h
 * \brief PPU-specific shared-memory layout variants for cutlass-ppu Actlize
 * layouts.
 */

#ifndef TL_PPU_LAYOUT_PPU_GEMM_LAYOUTS_H_
#define TL_PPU_LAYOUT_PPU_GEMM_LAYOUTS_H_

#include "layout/layout.h" // Layout, Buffer, etc.

#include <cstdint>

namespace tvm {
namespace tl {

// Register-fragment layouts used by PPU matrix loads.
Fragment MakeGemmFragment8x16PPU();
Fragment MakeGemmFragment8x32PPU(int dtype_bits);

// PPU: padded/shared-memory layout variants for cutlass-ppu Actlize layouts.
Layout MakeGemmBLayoutPaddedPPU(int stride, int continuous, int element_size,
                                bool k_inner = true);

Layout MakeGemmABLayoutPPU(int mat_stride, int mat_continuous, int continuity,
                           int element_size, bool k_inner = true,
                           bool is_gemm_rs = false);

// PPU: Buffer-based wrapper around MakeGemmABLayoutPPU.
Layout MakePPUSwizzledLayout(const tirx::Buffer &buffer, bool k_inner = true,
                             bool allow_pad = true, bool is_gemm_rs = false);

enum class AIUSharedLayoutClass : uint8_t {
  kAbsent,     // buffer has no entry in the layout map
  kLinear,     // present and semantically the row-major identity
  kSwizzle32,  // canonical 32B swizzle (unsupported by AIU)
  kSwizzle64,  // canonical 64B swizzle
  kSwizzle128, // canonical 128B swizzle
  kOther,      // present, non-identity, and not a canonical swizzle
};

AIUSharedLayoutClass ClassifySharedLayoutForAIU(const Layout &layout,
                                                const tirx::Buffer &buffer);

} // namespace tl
} // namespace tvm

#endif // TL_PPU_LAYOUT_PPU_GEMM_LAYOUTS_H_
