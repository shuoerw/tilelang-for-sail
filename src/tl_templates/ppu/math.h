#pragma once

#include "common.h"

#include <cutlass/fast_math.h>

#define hexp cutlass::fast_exp
#define hlog cutlass::fast_log
#define hsqrt cutlass::fast_sqrt
#define hsin cutlass::fast_sin
#define hcos cutlass::fast_cos

namespace cutlass {
// Mirror cutlass's own half_t fast_exp (fast_math.h): route through float.
// A direct `return ::hexp(x)` recurses, since `hexp` is #define'd to this
// same cutlass::fast_exp and x is already bfloat16_t.
TL_DEVICE
bfloat16_t fast_exp(bfloat16_t x) { return bfloat16_t(fast_exp(float(x))); }

// Same routing for fast_sqrt: the hsqrt macro below redirects every `hsqrt`
// call site (including the IEEE fsqrt lowering) to cutlass::fast_sqrt, which
// otherwise only resolves for float/double/half_t and leaves bfloat16_t
// results unassignable.
TL_DEVICE
bfloat16_t fast_sqrt(bfloat16_t x) { return bfloat16_t(fast_sqrt(float(x))); }
} // namespace cutlass

namespace tl {
TL_DEVICE float fast_rcp(float x) {
  float ret;
  asm volatile("ppu.rcp.approx.ftz.f32 %0, %1;" : "=f"(ret) : "f"(x));
  return ret;
}
} // namespace tl
