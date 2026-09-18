#pragma once

#ifndef __HGGCCC_RTC__
#include <hggc_runtime.h>
#endif

#include <cutlass/numeric_types.h>
#include <hggc/atomic>
#include <hggc_fp16.h>

using cutlass::bfloat16_t;
using cutlass::half_t;

#define TL_DEVICE __forceinline__ __device__
#define TL_NOT_IMPLEMENTED()                                                   \
  {                                                                            \
    printf("%s not implemented\n", __PRETTY_FUNCTION__);                       \
    __brkpt();                                                                 \
  }
template <typename T> struct normalize_atomic_type {
  using type = T;
};

template <> struct normalize_atomic_type<half_t> {
  using type = half;
};

template <> struct normalize_atomic_type<bfloat16_t> {
  using type = __ppu_bfloat16;
};

template <> struct normalize_atomic_type<int64_t> {
  using type = unsigned long long;
};

template <typename T1, typename T2> TL_DEVICE T1 hggc_cast(T2 val) {
  return T1(val);
}

template <> TL_DEVICE half hggc_cast<half, float>(float val) {
  return __float2half(val);
}

template <>
TL_DEVICE __ppu_bfloat16 hggc_cast<__ppu_bfloat16, float>(float val) {
  return __float2bfloat16(val);
}

// Helpers for atomic operations

namespace tl_atomic_detail {

TL_DEVICE bool IsRelaxedMemoryOrder(int memory_order) {
  return memory_order == int(hggc::memory_order_relaxed);
}

TL_DEVICE bool IsReleaseLikeMemoryOrder(int memory_order) {
  return memory_order == int(hggc::memory_order_release) ||
         memory_order == int(hggc::memory_order_consume);
}

TL_DEVICE bool IsAcquireMemoryOrder(int memory_order) {
  return memory_order == int(hggc::memory_order_acquire);
}

TL_DEVICE bool IsAcqRelLikeMemoryOrder(int memory_order) {
  return memory_order == int(hggc::memory_order_acq_rel) ||
         memory_order == int(hggc::memory_order_seq_cst);
}

template <typename T> TL_DEVICE unsigned short PackBits16(const T &val) {
  return *reinterpret_cast<const unsigned short *>(&val);
}

template <typename T> TL_DEVICE T UnpackBits16(unsigned short val) {
  return *reinterpret_cast<T *>(&val);
}

TL_DEVICE void tl_atomic_add_f16(unsigned short &ret, unsigned long long addr,
                                 unsigned short val, int memory_order) {
  if (IsReleaseLikeMemoryOrder(memory_order)) {
    asm volatile("ppu.atom.release.gpu.global.add.noftz.f16 %0, [%1], %2;"
                 : "=h"(ret)
                 : "l"(addr), "h"(val)
                 : "memory");
  } else if (IsAcquireMemoryOrder(memory_order)) {
    asm volatile("ppu.atom.acquire.gpu.global.add.noftz.f16 %0, [%1], %2;"
                 : "=h"(ret)
                 : "l"(addr), "h"(val)
                 : "memory");
  } else if (IsAcqRelLikeMemoryOrder(memory_order)) {
    asm volatile("ppu.atom.acq_rel.gpu.global.add.noftz.f16 %0, [%1], %2;"
                 : "=h"(ret)
                 : "l"(addr), "h"(val)
                 : "memory");
  }
}

TL_DEVICE void tl_atomic_add_bf16(unsigned short &ret, unsigned long long addr,
                                  unsigned short val, int memory_order) {
  if (IsReleaseLikeMemoryOrder(memory_order)) {
    asm volatile("ppu.atom.release.gpu.global.add.noftz.bf16 %0, [%1], %2;"
                 : "=h"(ret)
                 : "l"(addr), "h"(val)
                 : "memory");
  } else if (IsAcquireMemoryOrder(memory_order)) {
    asm volatile("ppu.atom.acquire.gpu.global.add.noftz.bf16 %0, [%1], %2;"
                 : "=h"(ret)
                 : "l"(addr), "h"(val)
                 : "memory");
  } else if (IsAcqRelLikeMemoryOrder(memory_order)) {
    asm volatile("ppu.atom.acq_rel.gpu.global.add.noftz.bf16 %0, [%1], %2;"
                 : "=h"(ret)
                 : "l"(addr), "h"(val)
                 : "memory");
  }
}

// Fallback implementations: do atomicAdd sequentially.

template <typename T> TL_DEVICE void AtomicAddx2Scalar(T *ref, T x, T y) {
  atomicAdd(ref + 0, x);
  atomicAdd(ref + 1, y);
}

template <typename T>
TL_DEVICE void AtomicAddx4Scalar(T *ref, T x, T y, T z, T w) {
  atomicAdd(ref + 0, x);
  atomicAdd(ref + 1, y);
  atomicAdd(ref + 2, z);
  atomicAdd(ref + 3, w);
}

TL_DEVICE float2 AtomicAddx2ScalarRet(float *ref, float2 add_val) {
  float2 ret;
  ret.x = atomicAdd(ref + 0, add_val.x);
  ret.y = atomicAdd(ref + 1, add_val.y);
  return ret;
}

template <typename VecT, typename T>
TL_DEVICE VecT AtomicAddx2ScalarRet(T *ref, VecT add_val) {
  VecT ret;
  ret.x = atomicAdd(ref + 0, add_val.x);
  ret.y = atomicAdd(ref + 1, add_val.y);
  return ret;
}

template <typename dst_dtype>
TL_DEVICE float4 AtomicAddx4ScalarRet(dst_dtype *ref, float4 add_val) {
  float4 ret;
  ret.x = atomicAdd(ref + 0, add_val.x);
  ret.y = atomicAdd(ref + 1, add_val.y);
  ret.z = atomicAdd(ref + 2, add_val.z);
  ret.w = atomicAdd(ref + 3, add_val.w);
  return ret;
}

} // namespace tl_atomic_detail

template <typename T1, typename T2>
TL_DEVICE void AtomicMax(T1 *ref, T2 val,
                         int memory_order = int(hggc::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = ref;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __ppu_bfloat16>) {
    // There is no implementation of atomicMax for half and bf16.
    // We simulate this process by atomicCAS loop.
    unsigned short *address_as_ushort =
        reinterpret_cast<unsigned short *>(address);
    unsigned short val_as_ushort = *reinterpret_cast<unsigned short *>(&val);
    unsigned short old_val_ushort = *address_as_ushort;
    while (val > *reinterpret_cast<T1 *>(&old_val_ushort)) {
      unsigned short assumed_val_ushort = old_val_ushort;
      old_val_ushort =
          atomicCAS(address_as_ushort, assumed_val_ushort, val_as_ushort);
      if (assumed_val_ushort == old_val_ushort) {
        break;
      }
    }
  } else {
    hggc::atomic_ref<NT1, hggc::thread_scope_device> aref(*address);
    aref.fetch_max(hggc_cast<NT1>(val), hggc::memory_order(memory_order));
  }
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicMaxRet(T1 *ref, T2 val,
                          int memory_order = int(hggc::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = ref;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __ppu_bfloat16>) {
    unsigned short *address_as_ushort =
        reinterpret_cast<unsigned short *>(address);
    unsigned short val_as_ushort = *reinterpret_cast<unsigned short *>(&val);
    unsigned short old_val_ushort = *address_as_ushort;
    while (val > *reinterpret_cast<T1 *>(&old_val_ushort)) {
      unsigned short assumed_val_ushort = old_val_ushort;
      old_val_ushort =
          atomicCAS(address_as_ushort, assumed_val_ushort, val_as_ushort);
      if (assumed_val_ushort == old_val_ushort) {
        break;
      }
    }
    return static_cast<T1>(*reinterpret_cast<T1 *>(&old_val_ushort));
  } else {
    hggc::atomic_ref<NT1, hggc::thread_scope_device> aref(*address);
    return static_cast<T1>(
        aref.fetch_max(hggc_cast<NT1>(val), hggc::memory_order(memory_order)));
  }
}

template <typename T1, typename T2>
TL_DEVICE void AtomicMin(T1 *ref, T2 val,
                         int memory_order = int(hggc::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = ref;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __ppu_bfloat16>) {
    // There is no implementation of atomicMin for half and bf16.
    // We simulate this process by atomicCAS loop.
    unsigned short *address_as_ushort =
        reinterpret_cast<unsigned short *>(address);
    unsigned short val_as_ushort = *reinterpret_cast<unsigned short *>(&val);
    unsigned short old_val_ushort = *address_as_ushort;
    while (val < *reinterpret_cast<T1 *>(&old_val_ushort)) {
      unsigned short assumed_val_ushort = old_val_ushort;
      old_val_ushort =
          atomicCAS(address_as_ushort, assumed_val_ushort, val_as_ushort);
      if (assumed_val_ushort == old_val_ushort) {
        break;
      }
    }
  } else {
    hggc::atomic_ref<NT1, hggc::thread_scope_device> aref(*address);
    aref.fetch_min(hggc_cast<NT1>(val), hggc::memory_order(memory_order));
  }
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicMinRet(T1 *ref, T2 val,
                          int memory_order = int(hggc::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  T1 *address = ref;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __ppu_bfloat16>) {
    unsigned short *address_as_ushort =
        reinterpret_cast<unsigned short *>(address);
    unsigned short val_as_ushort = *reinterpret_cast<unsigned short *>(&val);
    unsigned short old_val_ushort = *address_as_ushort;
    while (val < *reinterpret_cast<T1 *>(&old_val_ushort)) {
      unsigned short assumed_val_ushort = old_val_ushort;
      old_val_ushort =
          atomicCAS(address_as_ushort, assumed_val_ushort, val_as_ushort);
      if (assumed_val_ushort == old_val_ushort) {
        break;
      }
    }
    return static_cast<T1>(*reinterpret_cast<T1 *>(&old_val_ushort));
  } else {
    hggc::atomic_ref<NT1, hggc::thread_scope_device> aref(*address);
    return static_cast<T1>(
        aref.fetch_min(hggc_cast<NT1>(val), hggc::memory_order(memory_order)));
  }
}

template <typename T1, typename T2>
TL_DEVICE void AtomicAdd(T1 *address, T2 val,
                         int memory_order = int(hggc::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  (void)memory_order;
  atomicAdd(reinterpret_cast<NT1 *>(address), hggc_cast<NT1>(val));
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicAddRet(T1 *address, T2 val,
                          int memory_order = int(hggc::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  if constexpr (std::is_same_v<NT1, half> ||
                std::is_same_v<NT1, __ppu_bfloat16>) {
    if (tl_atomic_detail::IsRelaxedMemoryOrder(memory_order)) {
      return static_cast<T1>(
          atomicAdd(reinterpret_cast<NT1 *>(address), static_cast<NT1>(val)));
    } else {
      if constexpr (std::is_same_v<NT1, half>) {
        // fp16
        unsigned short ret_val_cast;
        unsigned long long ref_address =
            reinterpret_cast<unsigned long long>(address);
        unsigned short val_cast =
            tl_atomic_detail::PackBits16(hggc_cast<NT1>(val));
        tl_atomic_detail::tl_atomic_add_f16(ret_val_cast, ref_address, val_cast,
                                            memory_order);
        return static_cast<T1>(
            tl_atomic_detail::UnpackBits16<__half>(ret_val_cast));
      } else if constexpr (std::is_same_v<NT1, __ppu_bfloat16>) {
        // bf16
        unsigned short ret_val_cast;
        unsigned long long ref_address =
            reinterpret_cast<unsigned long long>(address);
        unsigned short val_cast =
            tl_atomic_detail::PackBits16(hggc_cast<NT1>(val));
        tl_atomic_detail::tl_atomic_add_bf16(ret_val_cast, ref_address,
                                             val_cast, memory_order);
        return static_cast<T1>(
            tl_atomic_detail::UnpackBits16<__ppu_bfloat16>(ret_val_cast));
      }
    }
  } else {
    hggc::atomic_ref<NT1, hggc::thread_scope_device> aref(*address);
    return static_cast<T1>(
        aref.fetch_add(hggc_cast<NT1>(val), hggc::memory_order(memory_order)));
  }
}

// For vectorized AtomicAdd, we maintain two versions of interfaces:
// 1. AtomicAddxN(dst_type* ref, src_type *val) // Pass pointer
// 2. AtomicAddxN(dst_type* ref, src_type val) // Pass value
template <typename T> TL_DEVICE half2 ToHalf2(T *val) {
  return *reinterpret_cast<const half2 *>(val);
}

template <typename T> TL_DEVICE half2 ToHalf2(T val) {
  return static_cast<half2>(*reinterpret_cast<const half2 *>(&val));
}

TL_DEVICE half2 ToHalf2(half2 val) { return val; }

// fp32 source: convert (round-to-nearest) instead of reinterpreting
TL_DEVICE half2 ToHalf2(float2 val) { return __float22half2_rn(val); }
TL_DEVICE half2 ToHalf2(const float *val) {
  return __float22half2_rn(make_float2(val[0], val[1]));
}
TL_DEVICE half2 ToHalf2(float *val) {
  return ToHalf2(static_cast<const float *>(val));
}

// Here ValType can be either value or value* (pointer)

template <typename ValType>
TL_DEVICE void AtomicAddx2(half_t *ref, ValType val,
                           int memory_order = int(hggc::memory_order_relaxed)) {
  half2 add_val = ToHalf2(val);
  if (tl_atomic_detail::IsRelaxedMemoryOrder(memory_order)) {
    atomicAdd(reinterpret_cast<half2 *>(ref), add_val);
  } else {
    // No vectorized atomic with memory order: fall back to scalar atomicAdd,
    // consistent with the f32 version.
    tl_atomic_detail::AtomicAddx2Scalar(reinterpret_cast<half *>(ref),
                                        static_cast<half>(add_val.x),
                                        static_cast<half>(add_val.y));
  }
}

template <typename ValType>
TL_DEVICE half2
AtomicAddx2Ret(half_t *ref, ValType val,
               int memory_order = int(hggc::memory_order_relaxed)) {
  half2 add_val = ToHalf2(val);
  if (tl_atomic_detail::IsRelaxedMemoryOrder(memory_order)) {
    return atomicAdd(reinterpret_cast<half2 *>(ref), add_val);
  } else {
    // No vectorized atomic with memory order: fall back to scalar atomicAdd,
    // consistent with the f32 version.
    return tl_atomic_detail::AtomicAddx2ScalarRet(reinterpret_cast<half *>(ref),
                                                  add_val);
  }
}

template <typename T> TL_DEVICE __ppu_bfloat162 ToBfloat162(T *val) {
  return *reinterpret_cast<const __ppu_bfloat162 *>(val);
}

template <typename T> TL_DEVICE __ppu_bfloat162 ToBfloat162(T val) {
  return static_cast<__ppu_bfloat162>(
      *reinterpret_cast<const __ppu_bfloat162 *>(&val));
}

TL_DEVICE __ppu_bfloat162 ToBfloat162(__ppu_bfloat162 val) { return val; }

// fp32 source: convert (round-to-nearest) instead of reinterpreting
TL_DEVICE __ppu_bfloat162 ToBfloat162(float2 val) {
  return __float22bfloat162_rn(val);
}
TL_DEVICE __ppu_bfloat162 ToBfloat162(const float *val) {
  return __float22bfloat162_rn(make_float2(val[0], val[1]));
}
TL_DEVICE __ppu_bfloat162 ToBfloat162(float *val) {
  return ToBfloat162(static_cast<const float *>(val));
}

template <typename ValType>
TL_DEVICE void AtomicAddx2(bfloat16_t *ref, ValType val,
                           int memory_order = int(hggc::memory_order_relaxed)) {
  __ppu_bfloat162 add_val = ToBfloat162(val);
  if (tl_atomic_detail::IsRelaxedMemoryOrder(memory_order)) {
    atomicAdd(reinterpret_cast<__ppu_bfloat162 *>(ref), add_val);
  } else {
    // No vectorized atomic with memory order: fall back to scalar atomicAdd,
    // consistent with the f32 version.
    tl_atomic_detail::AtomicAddx2Scalar(reinterpret_cast<__ppu_bfloat16 *>(ref),
                                        static_cast<__ppu_bfloat16>(add_val.x),
                                        static_cast<__ppu_bfloat16>(add_val.y));
  }
}

template <typename src_type>
TL_DEVICE __ppu_bfloat162
AtomicAddx2Ret(bfloat16_t *ref, src_type *val,
               int memory_order = int(hggc::memory_order_relaxed)) {
  if (tl_atomic_detail::IsRelaxedMemoryOrder(memory_order)) {
    return atomicAdd(reinterpret_cast<__ppu_bfloat162 *>(ref),
                     static_cast<__ppu_bfloat162>(
                         *reinterpret_cast<const __ppu_bfloat162 *>(val)));
  } else {
    __ppu_bfloat162 add_val = *reinterpret_cast<const __ppu_bfloat162 *>(val);
    // No vectorized atomic with memory order: fall back to scalar atomicAdd,
    // consistent with the f32 version.
    return tl_atomic_detail::AtomicAddx2ScalarRet(
        reinterpret_cast<__ppu_bfloat16 *>(ref), add_val);
  }
}

// PPU has no packed v4 f16/bf16 atomic add instruction; compose from the
// paired-lane version (same strategy as the CUDA pre-sm90 fallback).
template <typename SrcType>
TL_DEVICE void AtomicAddx4(half_t *ref, SrcType *val,
                           int memory_order = int(hggc::memory_order_relaxed)) {
  AtomicAddx2(ref, val, memory_order);
  AtomicAddx2(ref + 2, val + 2, memory_order);
}

template <typename SrcType>
TL_DEVICE void AtomicAddx4(bfloat16_t *ref, SrcType *val,
                           int memory_order = int(hggc::memory_order_relaxed)) {
  AtomicAddx2(ref, val, memory_order);
  AtomicAddx2(ref + 2, val + 2, memory_order);
}

template <typename T> TL_DEVICE float2 ToFloat2(T *val) {
  return *reinterpret_cast<const float2 *>(val);
}

TL_DEVICE float2 ToFloat2(float2 val) { return val; }

template <typename T> TL_DEVICE float4 ToFloat4(T *val) {
  return *reinterpret_cast<const float4 *>(val);
}

TL_DEVICE float4 ToFloat4(float4 val) { return val; }

template <typename ValType>
TL_DEVICE void AtomicAddx2(float *ref, ValType val,
                           int memory_order = int(hggc::memory_order_relaxed)) {
  (void)memory_order;
  float2 add_val = ToFloat2(val);
  tl_atomic_detail::AtomicAddx2Scalar(ref, add_val.x, add_val.y);
}

template <typename ValType>
TL_DEVICE float2
AtomicAddx2Ret(float *ref, ValType val,
               int memory_order = int(hggc::memory_order_relaxed)) {
  (void)memory_order;
  float2 add_val = ToFloat2(val);
  return tl_atomic_detail::AtomicAddx2ScalarRet(ref, add_val);
}

template <typename dst_dtype, typename ValType>
TL_DEVICE void AtomicAddx4(dst_dtype *ref, ValType val,
                           int memory_order = int(hggc::memory_order_relaxed)) {
  (void)memory_order;
  float4 add_val = ToFloat4(val);
  tl_atomic_detail::AtomicAddx4Scalar(ref, add_val.x, add_val.y, add_val.z,
                                      add_val.w);
}

template <typename dst_dtype, typename ValType>
TL_DEVICE float4
AtomicAddx4Ret(dst_dtype *ref, ValType val,
               int memory_order = int(hggc::memory_order_relaxed)) {
  (void)memory_order;
  float4 add_val = ToFloat4(val);
  return tl_atomic_detail::AtomicAddx4ScalarRet(ref, add_val);
}

template <hggc::thread_scope Scope = hggc::thread_scope_device, typename T>
TL_DEVICE T AtomicLoad(T *ref, int memory_order) {
  hggc::atomic_ref<T, Scope> aref(*ref);
  return aref.load(hggc::memory_order(memory_order));
}

template <hggc::thread_scope Scope = hggc::thread_scope_device, typename T1,
          typename T2>
TL_DEVICE void AtomicStore(T1 *ref, T2 value, int memory_order) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  hggc::atomic_ref<NT1, Scope> aref(*reinterpret_cast<NT1 *>(ref));
  aref.store(hggc_cast<NT1>(value), hggc::memory_order(memory_order));
}

template <typename T1, typename T2>
TL_DEVICE void PPUSystemAtomicAdd(T1 *address, T2 val, int memory_order) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  static_assert(std::is_same_v<NT1, int> || std::is_same_v<NT1, unsigned int>,
                "PPU system-scope atomic add supports int32 and uint32 only");
  if constexpr (std::is_same_v<NT1, int>) {
    int previous;
    if (memory_order == int(hggc::memory_order_relaxed)) {
      __ppu_atomic_fetch_add_relaxed_i32_system(
          reinterpret_cast<volatile int *>(address), previous,
          static_cast<int>(val));
    } else if (memory_order == int(hggc::memory_order_consume) ||
               memory_order == int(hggc::memory_order_acquire)) {
      __ppu_atomic_fetch_add_acquire_i32_system(
          reinterpret_cast<volatile int *>(address), previous,
          static_cast<int>(val));
    } else if (memory_order == int(hggc::memory_order_release)) {
      __ppu_atomic_fetch_add_release_i32_system(
          reinterpret_cast<volatile int *>(address), previous,
          static_cast<int>(val));
    } else if (memory_order == int(hggc::memory_order_acq_rel)) {
      __ppu_atomic_fetch_add_acq_rel_i32_system(
          reinterpret_cast<volatile int *>(address), previous,
          static_cast<int>(val));
    } else if (memory_order == int(hggc::memory_order_seq_cst)) {
      __ppu_fence_sc_system();
      __ppu_atomic_fetch_add_acquire_i32_system(
          reinterpret_cast<volatile int *>(address), previous,
          static_cast<int>(val));
    } else {
      TL_NOT_IMPLEMENTED();
    }
  } else {
    unsigned int previous;
    if (memory_order == int(hggc::memory_order_relaxed)) {
      __ppu_atomic_fetch_add_relaxed_u32_system(
          reinterpret_cast<volatile unsigned int *>(address), previous,
          static_cast<unsigned int>(val));
    } else if (memory_order == int(hggc::memory_order_consume) ||
               memory_order == int(hggc::memory_order_acquire)) {
      __ppu_atomic_fetch_add_acquire_u32_system(
          reinterpret_cast<volatile unsigned int *>(address), previous,
          static_cast<unsigned int>(val));
    } else if (memory_order == int(hggc::memory_order_release)) {
      __ppu_atomic_fetch_add_release_u32_system(
          reinterpret_cast<volatile unsigned int *>(address), previous,
          static_cast<unsigned int>(val));
    } else if (memory_order == int(hggc::memory_order_acq_rel)) {
      __ppu_atomic_fetch_add_acq_rel_u32_system(
          reinterpret_cast<volatile unsigned int *>(address), previous,
          static_cast<unsigned int>(val));
    } else if (memory_order == int(hggc::memory_order_seq_cst)) {
      __ppu_fence_sc_system();
      __ppu_atomic_fetch_add_acquire_u32_system(
          reinterpret_cast<volatile unsigned int *>(address), previous,
          static_cast<unsigned int>(val));
    } else {
      TL_NOT_IMPLEMENTED();
    }
  }
}

template <typename T>
TL_DEVICE T PPUSystemAtomicLoad(T *ref, int memory_order) {
  using NT = typename normalize_atomic_type<T>::type;
  static_assert(std::is_same_v<NT, int> || std::is_same_v<NT, unsigned int>,
                "PPU system-scope atomic load supports int32 and uint32 only");
  if constexpr (std::is_same_v<NT, int>) {
    int value;
    if (memory_order == int(hggc::memory_order_relaxed)) {
      __ppu_atomic_load_relaxed_i32_system(
          reinterpret_cast<const volatile int *>(ref), value);
      return value;
    }
    if (memory_order == int(hggc::memory_order_consume) ||
        memory_order == int(hggc::memory_order_acquire)) {
      __ppu_atomic_load_acquire_i32_system(
          reinterpret_cast<const volatile int *>(ref), value);
      return value;
    }
    if (memory_order == int(hggc::memory_order_seq_cst)) {
      __ppu_fence_sc_system();
      __ppu_atomic_load_acquire_i32_system(
          reinterpret_cast<const volatile int *>(ref), value);
      return value;
    }
  } else {
    unsigned int value;
    if (memory_order == int(hggc::memory_order_relaxed)) {
      __ppu_atomic_load_relaxed_u32_system(
          reinterpret_cast<const volatile unsigned int *>(ref), value);
      return value;
    }
    if (memory_order == int(hggc::memory_order_consume) ||
        memory_order == int(hggc::memory_order_acquire)) {
      __ppu_atomic_load_acquire_u32_system(
          reinterpret_cast<const volatile unsigned int *>(ref), value);
      return value;
    }
    if (memory_order == int(hggc::memory_order_seq_cst)) {
      __ppu_fence_sc_system();
      __ppu_atomic_load_acquire_u32_system(
          reinterpret_cast<const volatile unsigned int *>(ref), value);
      return value;
    }
  }
  TL_NOT_IMPLEMENTED();
  return T{};
}

template <typename T1, typename T2>
TL_DEVICE void PPUSystemAtomicStore(T1 *ref, T2 value, int memory_order) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  static_assert(std::is_same_v<NT1, int> || std::is_same_v<NT1, unsigned int>,
                "PPU system-scope atomic store supports int32 and uint32 only");
  if constexpr (std::is_same_v<NT1, int>) {
    if (memory_order == int(hggc::memory_order_relaxed)) {
      __ppu_atomic_store_relaxed_i32_system(
          reinterpret_cast<volatile int *>(ref), static_cast<int>(value));
      return;
    }
    if (memory_order == int(hggc::memory_order_release)) {
      __ppu_atomic_store_release_i32_system(
          reinterpret_cast<volatile int *>(ref), static_cast<int>(value));
      return;
    }
    if (memory_order == int(hggc::memory_order_seq_cst)) {
      __ppu_fence_sc_system();
      __ppu_atomic_store_relaxed_i32_system(
          reinterpret_cast<volatile int *>(ref), static_cast<int>(value));
      return;
    }
  } else if constexpr (std::is_same_v<NT1, unsigned int>) {
    if (memory_order == int(hggc::memory_order_relaxed)) {
      __ppu_atomic_store_relaxed_u32_system(
          reinterpret_cast<volatile unsigned int *>(ref),
          static_cast<unsigned int>(value));
      return;
    }
    if (memory_order == int(hggc::memory_order_release)) {
      __ppu_atomic_store_release_u32_system(
          reinterpret_cast<volatile unsigned int *>(ref),
          static_cast<unsigned int>(value));
      return;
    }
    if (memory_order == int(hggc::memory_order_seq_cst)) {
      __ppu_fence_sc_system();
      __ppu_atomic_store_relaxed_u32_system(
          reinterpret_cast<volatile unsigned int *>(ref),
          static_cast<unsigned int>(value));
      return;
    }
  }
  TL_NOT_IMPLEMENTED();
}

template <typename T1, typename T2>
TL_DEVICE void AtomicOr(T1 *ref, T2 value,
                        int memory_order = int(hggc::memory_order_relaxed)) {
  using NT1 = typename normalize_atomic_type<T1>::type;
  static_assert(std::is_integral_v<NT1>,
                "AtomicOr only supports integral types");
  hggc::atomic_ref<NT1, hggc::thread_scope_device> aref(*ref);
  aref.fetch_or(hggc_cast<NT1>(value), hggc::memory_order(memory_order));
}
