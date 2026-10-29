#pragma once

#ifndef __HGGCCC_RTC__
#include <cstdio>
#include <cstdlib>
#include <hggc_runtime.h>
#endif

#if !defined(__GNUC__) && !defined(__clang__)
#define __asm__ asm
#define __volatile__ volatile
#endif

#include "atomic.h"
#include <cute/arch/util.hpp>
#include <cutlass/numeric_types.h>
#include <hggc_math_constants.h>

#include <cutlass/bfloat16.h>
#include <cutlass/float8.h>

using cutlass::bfloat16_t;
using cutlass::half_t;

using cute::cast_smem_ptr_to_uint;

using int4_t = int4;

#define uint unsigned int
#define uchar unsigned char
#define ushort unsigned short

#define TL_DEVICE __forceinline__ __device__
#define TL_DEVICE_NOINLINE __noinline__ __device__
#define TL_PATCH

#define TILELANG_CHECK(stmt)                                                   \
  do {                                                                         \
    hggcError_t __err = (stmt);                                                \
    if (__err != hggcSuccess) {                                                \
      snprintf(error_buf, ERROR_BUF_SIZE, "%s:%d: %s - %s", __FILE__,          \
               __LINE__, hggcGetErrorName(__err), hggcGetErrorString(__err));  \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define TILELANG_CHECK_LAST_ERROR(kernel_name)                                 \
  do {                                                                         \
    hggcError_t __err = hggcGetLastError();                                    \
    if (__err != hggcSuccess) {                                                \
      snprintf(error_buf, ERROR_BUF_SIZE, kernel_name ": %s - %s",             \
               hggcGetErrorName(__err), hggcGetErrorString(__err));            \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#if defined(__HGGC_ARCH__)
#define TILELANG_UNREACHABLE(msg)                                              \
  do {                                                                         \
    printf("%s, %s:%d\n", msg, __FILE__, __LINE__);                            \
    __trap();                                                                  \
  } while (0)
#elif defined(__HGGCCC_RTC__)
#define TILELANG_UNREACHABLE(msg)                                              \
  do {                                                                         \
    __builtin_trap();                                                          \
  } while (0)
#else
#define TILELANG_UNREACHABLE(msg)                                              \
  do {                                                                         \
    fprintf(stderr, "%s, %s:%d\n", msg, __FILE__, __LINE__);                   \
    abort();                                                                   \
  } while (0)
#endif

// using cutlass abs function for half_t
TL_PATCH TL_DEVICE half_t __habs(const half_t x) {
  return half_t(__habs(x.to_half()));
}

// using cutlass abs function for bfloat_t
TL_PATCH TL_DEVICE bfloat16_t __habs(const bfloat16_t x) {
  return bfloat16_t(__habs(x.to_ppu_bfloat16()));
}

// hrsqrt function for half_t
TL_PATCH TL_DEVICE half_t hrsqrt(const half_t x) {
  return half_t(hrsqrt(x.to_half()));
}

// hrsqrt function for bfloat16_t
TL_PATCH TL_DEVICE bfloat16_t hrsqrt(const bfloat16_t x) {
  return bfloat16_t(hrsqrt(x.to_ppu_bfloat16()));
}

// IEEE-rounding half intrinsics emitted by PPUIEEEMath. The SDK declares
// them for __half / __ppu_bfloat16 only, so forward from the CUTLASS
// wrapper types.
TL_PATCH TL_DEVICE half_t __hadd_rn(const half_t x, const half_t y) {
  return half_t(__hadd_rn(x.to_half(), y.to_half()));
}

TL_PATCH TL_DEVICE bfloat16_t __hadd_rn(const bfloat16_t x,
                                        const bfloat16_t y) {
  return bfloat16_t(__hadd_rn(x.to_ppu_bfloat16(), y.to_ppu_bfloat16()));
}

TL_PATCH TL_DEVICE half_t __hsub_rn(const half_t x, const half_t y) {
  return half_t(__hsub_rn(x.to_half(), y.to_half()));
}

TL_PATCH TL_DEVICE bfloat16_t __hsub_rn(const bfloat16_t x,
                                        const bfloat16_t y) {
  return bfloat16_t(__hsub_rn(x.to_ppu_bfloat16(), y.to_ppu_bfloat16()));
}

TL_PATCH TL_DEVICE half_t __hmul_rn(const half_t x, const half_t y) {
  return half_t(__hmul_rn(x.to_half(), y.to_half()));
}

TL_PATCH TL_DEVICE bfloat16_t __hmul_rn(const bfloat16_t x,
                                        const bfloat16_t y) {
  return bfloat16_t(__hmul_rn(x.to_ppu_bfloat16(), y.to_ppu_bfloat16()));
}

TL_PATCH TL_DEVICE half_t __hdiv(const half_t x, const half_t y) {
  return half_t(__hdiv(x.to_half(), y.to_half()));
}

TL_PATCH TL_DEVICE bfloat16_t __hdiv(const bfloat16_t x, const bfloat16_t y) {
  return bfloat16_t(__hdiv(x.to_ppu_bfloat16(), y.to_ppu_bfloat16()));
}

TL_PATCH TL_DEVICE half_t __hfma(const half_t x, const half_t y,
                                 const half_t z) {
  return half_t(__hfma(x.to_half(), y.to_half(), z.to_half()));
}

TL_PATCH TL_DEVICE bfloat16_t __hfma(const bfloat16_t x, const bfloat16_t y,
                                     const bfloat16_t z) {
  return bfloat16_t(
      __hfma(x.to_ppu_bfloat16(), y.to_ppu_bfloat16(), z.to_ppu_bfloat16()));
}

TL_PATCH TL_DEVICE half_t hsqrt(const half_t x) {
  return half_t(hsqrt(x.to_half()));
}

TL_PATCH TL_DEVICE bfloat16_t hsqrt(const bfloat16_t x) {
  return bfloat16_t(hsqrt(x.to_ppu_bfloat16()));
}

TL_PATCH TL_DEVICE half_t hrcp(const half_t x) {
  return half_t(hrcp(x.to_half()));
}

TL_PATCH TL_DEVICE bfloat16_t hrcp(const bfloat16_t x) {
  return bfloat16_t(hrcp(x.to_ppu_bfloat16()));
}

TL_PATCH TL_DEVICE bfloat16_t hexp(const bfloat16_t x) {
  return bfloat16_t(hexp(x.to_ppu_bfloat16()));
}

// 16-bit lowering emits these h* names, but PPU has no overload of them
// that accepts the CUTLASS wrapper types. Evaluate in float32 and convert
// the result back to the source type.
TL_PATCH TL_DEVICE half_t hsinh(const half_t x) {
  return half_t(sinhf(float(x)));
}

TL_PATCH TL_DEVICE bfloat16_t hsinh(const bfloat16_t x) {
  return bfloat16_t(sinhf(float(x)));
}

TL_PATCH TL_DEVICE half_t hcosh(const half_t x) {
  return half_t(coshf(float(x)));
}

TL_PATCH TL_DEVICE bfloat16_t hcosh(const bfloat16_t x) {
  return bfloat16_t(coshf(float(x)));
}

TL_PATCH TL_DEVICE half_t htanh(const half_t x) {
  return half_t(tanhf(float(x)));
}

TL_PATCH TL_DEVICE bfloat16_t htanh(const bfloat16_t x) {
  return bfloat16_t(tanhf(float(x)));
}

TL_PATCH TL_DEVICE half_t hatan(const half_t x) {
  return half_t(atanf(float(x)));
}

TL_PATCH TL_DEVICE bfloat16_t hatan(const bfloat16_t x) {
  return bfloat16_t(atanf(float(x)));
}

TL_PATCH TL_DEVICE half_t herf(const half_t x) {
  return half_t(erff(float(x)));
}

TL_PATCH TL_DEVICE bfloat16_t herf(const bfloat16_t x) {
  return bfloat16_t(erff(float(x)));
}

TL_PATCH TL_DEVICE half_t hnearbyint(const half_t x) {
  return half_t(nearbyintf(float(x)));
}

TL_PATCH TL_DEVICE bfloat16_t hnearbyint(const bfloat16_t x) {
  return bfloat16_t(nearbyintf(float(x)));
}

TL_PATCH TL_DEVICE half_t hpow(const half_t x, const half_t y) {
  return half_t(powf(float(x), float(y)));
}

TL_PATCH TL_DEVICE bfloat16_t hpow(const bfloat16_t x, const bfloat16_t y) {
  return bfloat16_t(powf(float(x), float(y)));
}

TL_PATCH TL_DEVICE half_t hfmod(const half_t x, const half_t y) {
  return half_t(fmodf(float(x), float(y)));
}

TL_PATCH TL_DEVICE bfloat16_t hfmod(const bfloat16_t x, const bfloat16_t y) {
  return bfloat16_t(fmodf(float(x), float(y)));
}

// Pack two half values.
TL_DEVICE unsigned __pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack two half_t values.
TL_DEVICE unsigned __pack_half2(const half_t x, const half_t y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack two bfloat16_t values.
TL_DEVICE unsigned __pack_half2(const bfloat16_t x, const bfloat16_t y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack two bfloat16_t values.
TL_DEVICE unsigned __pack_ppu_bfloat162(const bfloat16_t x,
                                        const bfloat16_t y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack four char values.
TL_DEVICE int make_int(signed char x0, signed char x1, signed char x2,
                       signed char x3) {
  const unsigned int b0 = static_cast<unsigned char>(x0);
  const unsigned int b1 = static_cast<unsigned char>(x1);
  const unsigned int b2 = static_cast<unsigned char>(x2);
  const unsigned int b3 = static_cast<unsigned char>(x3);
  return static_cast<int>((b3 << 24) | (b2 << 16) | (b1 << 8) | b0);
}

// Pack eight char values.
TL_DEVICE int2 make_int2(signed char x0, signed char x1, signed char x2,
                         signed char x3, signed char y0, signed char y1,
                         signed char y2, signed char y3) {
  int2 result;
  result.x = make_int(x0, x1, x2, x3);
  result.y = make_int(y0, y1, y2, y3);
  return result;
}

// Pack sixteen char values.
TL_DEVICE int4_t make_int4(signed char x0, signed char x1, signed char x2,
                           signed char x3, signed char y0, signed char y1,
                           signed char y2, signed char y3, signed char z0,
                           signed char z1, signed char z2, signed char z3,
                           signed char w0, signed char w1, signed char w2,
                           signed char w3) {
  int4_t result;
  result.x = make_int(x0, x1, x2, x3);
  result.y = make_int(y0, y1, y2, y3);
  result.z = make_int(z0, z1, z2, z3);
  result.w = make_int(w0, w1, w2, w3);
  return result;
}

TL_DEVICE int4_t make_int4(short x0, short x1, short y0, short y1, short z0,
                           short z1, short w0, short w1) {
  int4_t result;
  *((short2 *)&result.x) = make_short2(x0, x1);
  *((short2 *)&result.y) = make_short2(y0, y1);
  *((short2 *)&result.z) = make_short2(z0, z1);
  *((short2 *)&result.w) = make_short2(w0, w1);
  return result;
}

// Pack four char values.
TL_DEVICE unsigned int make_uint(unsigned char x0, unsigned char x1,
                                 unsigned char x2, unsigned char x3) {
  return (x3 << 24) | (x2 << 16) | (x1 << 8) | x0;
}

// Pack eight char values.
TL_DEVICE uint2 make_uint2(unsigned char x0, unsigned char x1, unsigned char x2,
                           unsigned char x3, unsigned char y0, unsigned char y1,
                           unsigned char y2, unsigned char y3) {
  uint2 result;
  result.x = make_uint(x0, x1, x2, x3);
  result.y = make_uint(y0, y1, y2, y3);
  return result;
}

// Pack sixteen char values.
TL_DEVICE uint4 make_uint4(unsigned char x0, unsigned char x1, unsigned char x2,
                           unsigned char x3, unsigned char y0, unsigned char y1,
                           unsigned char y2, unsigned char y3, unsigned char z0,
                           unsigned char z1, unsigned char z2, unsigned char z3,
                           unsigned char w0, unsigned char w1, unsigned char w2,
                           unsigned char w3) {
  uint4 result;
  result.x = make_uint(x0, x1, x2, x3);
  result.y = make_uint(y0, y1, y2, y3);
  result.z = make_uint(z0, z1, z2, z3);
  result.w = make_uint(w0, w1, w2, w3);
  return result;
}

TL_DEVICE uint4 make_uint4(unsigned short x0, unsigned short x1,
                           unsigned short y0, unsigned short y1,
                           unsigned short z0, unsigned short z1,
                           unsigned short w0, unsigned short w1) {
  uint4 result;
  *((ushort2 *)&result.x) = make_ushort2(x0, x1);
  *((ushort2 *)&result.y) = make_ushort2(y0, y1);
  *((ushort2 *)&result.z) = make_ushort2(z0, z1);
  *((ushort2 *)&result.w) = make_ushort2(w0, w1);
  return result;
}

// ============================================================================
// Packed INT4 Buffer Access Helpers
// ============================================================================
// TileLang lowers scalar int4/uint4 storage through byte-packed buffers, where
// each byte carries 2 logical 4-bit elements.

TL_DEVICE int tl_int4_packed_load(const signed char *packed, int idx) {
  unsigned char byte = static_cast<unsigned char>(packed[idx >> 1]);
  unsigned int shift = (idx & 1) * 4;
  int value = static_cast<int>((byte >> shift) & 0xF);
  return (value << 28) >> 28;
}

TL_DEVICE unsigned int tl_uint4_packed_load(const unsigned char *packed,
                                            int idx) {
  unsigned char byte = packed[idx >> 1];
  unsigned int shift = (idx & 1) * 4;
  return (byte >> shift) & 0xF;
}

TL_DEVICE void tl_int4_packed_store(signed char *packed, int idx, int val) {
  unsigned int shift = (idx & 1) * 4;
  unsigned char mask = static_cast<unsigned char>(0xFu << shift);
  unsigned char nibble = static_cast<unsigned char>(
      (static_cast<unsigned int>(val) & 0xF) << shift);
  unsigned char byte = static_cast<unsigned char>(packed[idx >> 1]);
  packed[idx >> 1] = static_cast<signed char>((byte & ~mask) | nibble);
}

TL_DEVICE void tl_uint4_packed_store(unsigned char *packed, int idx,
                                     unsigned int val) {
  unsigned int shift = (idx & 1) * 4;
  unsigned char mask = static_cast<unsigned char>(0xFu << shift);
  unsigned char nibble = static_cast<unsigned char>((val & 0xF) << shift);
  packed[idx >> 1] =
      static_cast<unsigned char>((packed[idx >> 1] & ~mask) | nibble);
}

// Pack eight int values.
TL_DEVICE longlong4 make_longlong4(int x0, int x1, int y0, int y1, int z0,
                                   int z1, int w0, int w1) {
  longlong4 result;
  *((int2 *)&result.x) = make_int2(x0, x1);
  *((int2 *)&result.y) = make_int2(y0, y1);
  *((int2 *)&result.z) = make_int2(z0, z1);
  *((int2 *)&result.w) = make_int2(w0, w1);
  return result;
}

// Helper to cast SMEM pointer to unsigned
TL_DEVICE uint32_t smem_ptr_to_uint(void const *const ptr) {
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

/**
 * Convert a shared-memory pointer to a 32-bit unsigned integer address.
 *
 * Casts the given pointer (expected to reference shared memory) into a 32-bit
 * unsigned integer using the device address-space conversion required for
 * shared-memory pointers.
 *
 * @param smem_ptr Pointer into shared memory.
 * @return 32-bit unsigned integer representation of the shared-memory address.
 *
 * @note The pointer must refer to shared memory; behavior is undefined for
 *       pointers in other address spaces.
 */
TL_DEVICE unsigned int cast_smem_ptr_to_int(const void *const smem_ptr) {
  unsigned int smem_int;
  asm volatile("{ .reg .u64 smem_int; ppu.cvta.to.shared.u64 smem_int, %1; "
               "ppu.cvt.u32.u64 %0, smem_int; }"
               : "=r"(smem_int)
               : "l"(smem_ptr));
  return smem_int;
}

// DP4A
template <typename InDatatype, typename OutDatatype>
TL_DEVICE /**
           * Compute a 4×8-bit dot-product-accumulate using the HGGC DP4A
           * intrinsic.
           *
           * Reads 32-bit packed values from `a` and `b` (each containing four
           * signed 8-bit lanes), applies the __dp4a operation (dot product of
           * the four lane pairs added to an accumulator), and stores the 32-bit
           * integer result through `c`.
           *
           * @param a Pointer to a 32-bit packed input containing four signed
           * 8-bit elements.
           * @param b Pointer to a 32-bit packed input containing four signed
           * 8-bit elements.
           * @param c Pointer to a 32-bit accumulator; its current value is used
           * as the initial accumulator and overwritten with the resulting int32
           * sum.
           */
    void
    DP4A(InDatatype *a, InDatatype *b, OutDatatype *c) {
  const int a_int = *((int *)a);
  const int b_int = *((int *)b);
  const int c_int = *((int *)c);
  *c = __dp4a(a_int, b_int, c_int);
}

namespace tl {

enum class DataType : int {
  kInt4 = 0,
  kUInt4 = 1,
  kInt8 = 2,
  kUInt8 = 3,
  kInt16 = 4,
  kUInt16 = 5,
  kInt32 = 6,
  kUInt32 = 7,
  kInt64 = 8,
  kUInt64 = 9,
  kFloat8_e4m3 = 10,
  kFloat8_e5m2 = 11,
  kFloat16 = 12,
  kBFloat16 = 13,
  kFloat16x2 = 14,
  kFloat32 = 15,
  kTensorFloat32 = 16,
  kFloat64 = 17,
  kBit1 = 18,
  kBit8 = 19,
  kBit16 = 20,
  kBit32 = 21,
  kBit64 = 22,
  kFloat6_e2m3fn = 23,
  kFloat6_e3m2fn = 24,
  kFloat4_e2m1fn = 25
};

// Any
template <typename T> TL_DEVICE bool Any(T *a, int size) {
  for (int i = 0; i < size; i++) {
    if (a[i]) {
      return true;
    }
  }
  return false;
}

// All
template <typename T> TL_DEVICE bool All(T *a, int size) {
  for (int i = 0; i < size; i++) {
    if (!a[i]) {
      return false;
    }
  }
  return true;
}

// Pow of int
template <int y = 1, typename T> TL_DEVICE T pow_of_int(T x) {
  T result = x;
  for (int i = 1; i < y; i++) {
    result *= x;
  }
  return result;
}

// Pack four 8-bit payloads (reinterpreted as raw bytes) into a 32-bit word.
template <typename T> TL_DEVICE unsigned int pack_b8x4(T x0, T x1, T x2, T x3) {
  return make_uint(*reinterpret_cast<unsigned char *>(&x0),
                   *reinterpret_cast<unsigned char *>(&x1),
                   *reinterpret_cast<unsigned char *>(&x2),
                   *reinterpret_cast<unsigned char *>(&x3));
}

// Find the position of the offset-th set bit in `mask` relative to `base`.
// Positive offsets scan upward starting at (and including) `base`;
// negative offsets scan downward starting just below `base`.
// Returns 0xFFFFFFFF when no such bit exists.
TL_DEVICE unsigned int fns(unsigned int mask, unsigned int base, int offset) {
  if (offset > 0) {
    for (unsigned int pos = base; pos < 32; ++pos) {
      if (mask & (1U << pos)) {
        if (--offset == 0) {
          return pos;
        }
      }
    }
  } else if (offset < 0) {
    for (int pos = static_cast<int>(base) - 1; pos >= 0; --pos) {
      if (mask & (1U << pos)) {
        if (++offset == 0) {
          return static_cast<unsigned int>(pos);
        }
      }
    }
  }
  return 0xFFFFFFFFU;
}

// Thread partial barrier synchronization
TL_DEVICE void __sync_thread_partial(int barrier_id = 0, int thread_count = 0) {
  asm volatile("ppu.bar.sync %0, %1;" : : "r"(barrier_id), "r"(thread_count));
}

// CTA named barrier one-sided arrive (bar.arrive).
// Signals arrival at the named barrier without waiting for other participants.
// Useful in warp-specialized pipelines where one warp group signals readiness
// without blocking, while the other waits with bar.sync /
// __sync_thread_partial.
TL_DEVICE void __named_barrier_arrive(int barrier_id, int thread_count) {
  asm volatile("ppu.bar.arrive %0, %1;" : : "r"(barrier_id), "r"(thread_count));
}

// and add the desired implicit conversion from bfloat16_t.
struct float_e4m3_t : public cute::float_e4m3_t {
  using cute::float_e4m3_t::float_e4m3_t;
  CUTLASS_HOST_DEVICE
  float_e4m3_t() = default;

  CUTLASS_HOST_DEVICE
  explicit float_e4m3_t(__ppu_bfloat16 x)
      : float_e4m3_t(static_cast<float>(x)) {}

  CUTLASS_HOST_DEVICE
  float_e4m3_t(cutlass::float_e4m3_t x)
      : cute::float_e4m3_t(*reinterpret_cast<cute::float_e4m3_t *>(&x)) {}
};

struct float_e5m2_t : public cute::float_e5m2_t {
  using cute::float_e5m2_t::float_e5m2_t;
  CUTLASS_HOST_DEVICE
  float_e5m2_t() = default;

  CUTLASS_HOST_DEVICE
  explicit float_e5m2_t(__ppu_bfloat16 x)
      : float_e5m2_t(static_cast<float>(x)) {}

  CUTLASS_HOST_DEVICE
  float_e5m2_t(cutlass::float_e5m2_t x)
      : cute::float_e5m2_t(*reinterpret_cast<cute::float_e5m2_t *>(&x)) {}
};

struct tfloat32_t : public cute::tfloat32_t {
  using cute::tfloat32_t::tfloat32_t;
  CUTLASS_HOST_DEVICE
  tfloat32_t() = default;

  CUTLASS_HOST_DEVICE
  explicit tfloat32_t(__ppu_bfloat16 x) : tfloat32_t(static_cast<float>(x)) {}

  CUTLASS_HOST_DEVICE
  tfloat32_t(cutlass::tfloat32_t x)
      : cute::tfloat32_t(*reinterpret_cast<cute::tfloat32_t *>(&x)) {}
};

template <typename T> struct to_cute_type {
  using type = T;
};
template <> struct to_cute_type<tl::float_e4m3_t> {
  using type = cute::float_e4m3_t;
};
template <> struct to_cute_type<tl::float_e5m2_t> {
  using type = cute::float_e5m2_t;
};
template <> struct to_cute_type<tl::tfloat32_t> {
  using type = cute::tfloat32_t;
};

// =========================================================================
// Packed x2 element-wise math helpers
//
// Each operation (add2, sub2, mul2, fma2, max2, min2, abs2) is provided for
// three dtype families:
//   1. float2           (FP32x2)
//   2. __ppu_bfloat162   (BF16x2)
//   3. __half2          (FP16x2)
//
// TVM stores bfloat16x2 and float16x2 as ``uint1`` in generated HGGC code.
// The HGGC codegen emits explicit casts from uint1 to __ppu_bfloat162 or
// __half2 based on the TIR dtype, so C++ overload resolution correctly
// dispatches to the right overload without ambiguous uint1 bridges.
// =========================================================================

// Cast helpers between uint1 and native packed types.
// Used by the HGGC codegen to convert between TVM's uint1 representation
// and the native __ppu_bfloat162 / __half2 types.
template <typename T> TL_DEVICE T from_uint1(uint1 v) {
  T r;
  memcpy(&r, &v, sizeof(T));
  return r;
}

template <typename T> TL_DEVICE uint1 to_uint1(T v) {
  uint1 r;
  memcpy(&r, &v, sizeof(uint1));
  return r;
}

// Pack two half_t into a uint1.
TL_DEVICE uint1 pack_half2(half_t a, half_t b) {
  unsigned packed =
      __pack_half2(static_cast<__half>(a), static_cast<__half>(b));
  return uint1{packed};
}

// --- add2 ----------------------------------------------------------------

TL_DEVICE float2 add2(float2 a, float2 b) {
  return make_float2(a.x + b.x, a.y + b.y);
}

TL_DEVICE __ppu_bfloat162 add2(__ppu_bfloat162 a, __ppu_bfloat162 b) {
  return __hadd2(a, b);
}

TL_DEVICE __half2 add2(__half2 a, __half2 b) { return __hadd2(a, b); }

// Note: uint1 bridge overloads removed -- the HGGC codegen now emits
// explicit casts to __ppu_bfloat162 or __half2 based on the TIR dtype,
// so C++ overload resolution correctly dispatches to the right overload.

// --- sub2 ----------------------------------------------------------------

TL_DEVICE float2 sub2(float2 a, float2 b) {
  return make_float2(a.x - b.x, a.y - b.y);
}

TL_DEVICE __ppu_bfloat162 sub2(__ppu_bfloat162 a, __ppu_bfloat162 b) {
  return __hsub2(a, b);
}

TL_DEVICE __half2 sub2(__half2 a, __half2 b) { return __hsub2(a, b); }

// --- mul2 ----------------------------------------------------------------

TL_DEVICE float2 mul2(float2 a, float2 b) {
  return make_float2(a.x * b.x, a.y * b.y);
}

TL_DEVICE __ppu_bfloat162 mul2(__ppu_bfloat162 a, __ppu_bfloat162 b) {
  return __hmul2(a, b);
}

TL_DEVICE __half2 mul2(__half2 a, __half2 b) { return __hmul2(a, b); }

// --- fma2 ----------------------------------------------------------------

TL_DEVICE float2 fma2(float2 a, float2 b, float2 c) {
  return make_float2(a.x * b.x + c.x, a.y * b.y + c.y);
}

TL_DEVICE __ppu_bfloat162 fma2(__ppu_bfloat162 a, __ppu_bfloat162 b,
                               __ppu_bfloat162 c) {
  return __hfma2(a, b, c);
}

TL_DEVICE __half2 fma2(__half2 a, __half2 b, __half2 c) {
  return __hfma2(a, b, c);
}

// --- fast_max / fast_min -------------------------------------------------

template <typename T> TL_DEVICE T fast_max(T a, T b) { return a < b ? b : a; }

template <> TL_DEVICE float fast_max(float a, float b) { return fmaxf(a, b); }

template <typename T> TL_DEVICE T fast_min(T a, T b) { return b < a ? b : a; }

template <> TL_DEVICE float fast_min(float a, float b) { return fminf(a, b); }

// --- max2 ----------------------------------------------------------------

TL_DEVICE float2 max2(float2 a, float2 b) {
  return make_float2(fmaxf(a.x, b.x), fmaxf(a.y, b.y));
}

TL_DEVICE __ppu_bfloat162 max2(__ppu_bfloat162 a, __ppu_bfloat162 b) {
  return __hmax2(a, b);
}

TL_DEVICE __half2 max2(__half2 a, __half2 b) { return __hmax2(a, b); }

// --- min2 ----------------------------------------------------------------

TL_DEVICE float2 min2(float2 a, float2 b) {
  return make_float2(fminf(a.x, b.x), fminf(a.y, b.y));
}

TL_DEVICE __ppu_bfloat162 min2(__ppu_bfloat162 a, __ppu_bfloat162 b) {
  return __hmin2(a, b);
}

TL_DEVICE __half2 min2(__half2 a, __half2 b) { return __hmin2(a, b); }

// --- max2_nan ------------------------------------------------------------

TL_DEVICE __ppu_bfloat162 max2_nan(__ppu_bfloat162 a, __ppu_bfloat162 b) {
  return __hmax2_nan(a, b);
}

TL_DEVICE __half2 max2_nan(__half2 a, __half2 b) { return __hmax2_nan(a, b); }

// --- min2_nan ------------------------------------------------------------

TL_DEVICE __ppu_bfloat162 min2_nan(__ppu_bfloat162 a, __ppu_bfloat162 b) {
  return __hmin2_nan(a, b);
}

TL_DEVICE __half2 min2_nan(__half2 a, __half2 b) { return __hmin2_nan(a, b); }

// --- abs2 ----------------------------------------------------------------

TL_DEVICE float2 abs2(float2 a) { return make_float2(fabsf(a.x), fabsf(a.y)); }

TL_DEVICE __ppu_bfloat162 abs2(__ppu_bfloat162 a) { return __habs2(a); }

TL_DEVICE __half2 abs2(__half2 a) { return __habs2(a); }

} // namespace tl

using tl::tfloat32_t;

//
// Optimized type-punned warp shuffle helpers for 16-bit types
// Directly shuffle the underlying bits (as uint16/uint32) to avoid
// costly fp32 conversions and instruction overhead.
//
namespace tl {

// Generic passthroughs
template <typename T>
TL_DEVICE T shfl_xor_sync(unsigned mask, T val, int laneMask) {
  return __shfl_xor_sync(mask, val, laneMask);
}

template <typename T>
TL_DEVICE T shfl_down_sync(unsigned mask, T val, int delta) {
  return __shfl_down_sync(mask, val, delta);
}

template <typename T>
TL_DEVICE T shfl_up_sync(unsigned mask, T val, int delta) {
  return __shfl_up_sync(mask, val, delta);
}

template <typename T> TL_DEVICE T shfl_sync(unsigned mask, T val, int srcLane) {
  return __shfl_sync(mask, val, srcLane);
}

// Specializations for cutlass::half_t
template <>
TL_DEVICE half_t shfl_xor_sync(unsigned mask, half_t val, int laneMask) {
  uint16_t raw = reinterpret_cast<uint16_t &>(val);
  uint32_t raw32 = static_cast<uint32_t>(raw);
  uint32_t ret32 = __shfl_xor_sync(mask, raw32, laneMask);
  uint16_t ret16 = static_cast<uint16_t>(ret32);
  return reinterpret_cast<half_t &>(ret16);
}

template <>
TL_DEVICE half_t shfl_down_sync(unsigned mask, half_t val, int delta) {
  uint16_t raw = reinterpret_cast<uint16_t &>(val);
  uint32_t raw32 = static_cast<uint32_t>(raw);
  uint32_t ret32 = __shfl_down_sync(mask, raw32, delta);
  uint16_t ret16 = static_cast<uint16_t>(ret32);
  return reinterpret_cast<half_t &>(ret16);
}

template <>
TL_DEVICE half_t shfl_up_sync(unsigned mask, half_t val, int delta) {
  uint16_t raw = reinterpret_cast<uint16_t &>(val);
  uint32_t raw32 = static_cast<uint32_t>(raw);
  uint32_t ret32 = __shfl_up_sync(mask, raw32, delta);
  uint16_t ret16 = static_cast<uint16_t>(ret32);
  return reinterpret_cast<half_t &>(ret16);
}

template <> TL_DEVICE half_t shfl_sync(unsigned mask, half_t val, int srcLane) {
  uint16_t raw = reinterpret_cast<uint16_t &>(val);
  uint32_t raw32 = static_cast<uint32_t>(raw);
  uint32_t ret32 = __shfl_sync(mask, raw32, srcLane);
  uint16_t ret16 = static_cast<uint16_t>(ret32);
  return reinterpret_cast<half_t &>(ret16);
}

// Specializations for cutlass::bfloat16_t
template <>
TL_DEVICE bfloat16_t shfl_xor_sync(unsigned mask, bfloat16_t val,
                                   int laneMask) {
  uint16_t raw = reinterpret_cast<uint16_t &>(val);
  uint32_t raw32 = static_cast<uint32_t>(raw);
  uint32_t ret32 = __shfl_xor_sync(mask, raw32, laneMask);
  uint16_t ret16 = static_cast<uint16_t>(ret32);
  return reinterpret_cast<bfloat16_t &>(ret16);
}

template <>
TL_DEVICE bfloat16_t shfl_down_sync(unsigned mask, bfloat16_t val, int delta) {
  uint16_t raw = reinterpret_cast<uint16_t &>(val);
  uint32_t raw32 = static_cast<uint32_t>(raw);
  uint32_t ret32 = __shfl_down_sync(mask, raw32, delta);
  uint16_t ret16 = static_cast<uint16_t>(ret32);
  return reinterpret_cast<bfloat16_t &>(ret16);
}

template <>
TL_DEVICE bfloat16_t shfl_up_sync(unsigned mask, bfloat16_t val, int delta) {
  uint16_t raw = reinterpret_cast<uint16_t &>(val);
  uint32_t raw32 = static_cast<uint32_t>(raw);
  uint32_t ret32 = __shfl_up_sync(mask, raw32, delta);
  uint16_t ret16 = static_cast<uint16_t>(ret32);
  return reinterpret_cast<bfloat16_t &>(ret16);
}

template <>
TL_DEVICE bfloat16_t shfl_sync(unsigned mask, bfloat16_t val, int srcLane) {
  uint16_t raw = reinterpret_cast<uint16_t &>(val);
  uint32_t raw32 = static_cast<uint32_t>(raw);
  uint32_t ret32 = __shfl_sync(mask, raw32, srcLane);
  uint16_t ret16 = static_cast<uint16_t>(ret32);
  return reinterpret_cast<bfloat16_t &>(ret16);
}

// Specializations for uint1 (packed bfloat16x2 / float16x2).
// uint1 is a 32-bit struct { unsigned x; } used to represent packed pairs.
// __shfl_xor_sync operates on native 32-bit types, so we pass the raw unsigned.

template <>
TL_DEVICE uint1 shfl_xor_sync(unsigned mask, uint1 val, int laneMask) {
  return uint1{__shfl_xor_sync(mask, val.x, laneMask)};
}

template <>
TL_DEVICE uint1 shfl_down_sync(unsigned mask, uint1 val, int delta) {
  return uint1{__shfl_down_sync(mask, val.x, delta)};
}

template <> TL_DEVICE uint1 shfl_up_sync(unsigned mask, uint1 val, int delta) {
  return uint1{__shfl_up_sync(mask, val.x, delta)};
}

template <> TL_DEVICE uint1 shfl_sync(unsigned mask, uint1 val, int srcLane) {
  return uint1{__shfl_sync(mask, val.x, srcLane)};
}

} // namespace tl
