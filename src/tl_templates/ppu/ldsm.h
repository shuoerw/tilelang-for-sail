#pragma once

#include "common.h"

namespace tl {

TL_DEVICE void tix_ldmatrix_x1(void const *const smem_ptr,
                               void *const local_ptr) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  int32_t *value = reinterpret_cast<int32_t *>(local_ptr);
#if defined(__HGGC_ARCH__)
  asm volatile(
#if __HGGC_ARCH__ == 100
      "ppu.tc01.ex.ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
#elif __HGGC_ARCH__ == 150
      "ppu.tc02.ldmatrix.sync.aligned.x1.m8n8.shared.b16 {%0}, [%1];\n"
#endif
      : "=r"(value[0])
      : "r"(smem_int_ptr));
#endif
}

TL_DEVICE void tix_ldmatrix_x2(void const *const smem_ptr,
                               void *const local_ptr) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  int32_t *value = reinterpret_cast<int32_t *>(local_ptr);
#if defined(__HGGC_ARCH__)
  asm volatile(
#if __HGGC_ARCH__ == 100
      "ppu.tc01.ex.ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
#elif __HGGC_ARCH__ == 150
      "ppu.tc02.ldmatrix.sync.aligned.x2.m8n8.shared.b16 {%0, %1}, [%2];\n"
#endif
      : "=r"(value[0]), "=r"(value[1])
      : "r"(smem_int_ptr));
#endif
}

TL_DEVICE void tix_ldmatrix_x4(void const *const smem_ptr,
                               void *const local_ptr) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  int32_t *value = reinterpret_cast<int32_t *>(local_ptr);
#if defined(__HGGC_ARCH__)
  asm volatile(
#if __HGGC_ARCH__ == 100
      "ppu.tc01.ex.ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0, %1, %2, %3}, "
      "[%4];\n"
#elif __HGGC_ARCH__ == 150
      "ppu.tc02.ldmatrix.sync.aligned.x4.m8n8.shared.b16 {%0, %1, %2, %3}, "
      "[%4];\n"
#endif
      : "=r"(value[0]), "=r"(value[1]), "=r"(value[2]), "=r"(value[3])
      : "r"(smem_int_ptr));
#endif
}

TL_DEVICE void tix_ldmatrix_swzl_x4(void const *const smem_ptr,
                                    void *const local_ptr, uint32_t mode,
                                    bool trans_block) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr) / 16;
  int32_t *value = reinterpret_cast<int32_t *>(local_ptr);
  if (!trans_block) {
    asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.shared.b16 "
                 "{%0, %1, %2, %3}, [%4], 1, %5, %6;\n"
                 : "=r"(value[0]), "=r"(value[1]), "=r"(value[2]),
                   "=r"(value[3])
                 : "r"(smem_int_ptr), "r"(64 - mode * 32), "r"(mode));
  } else {
    asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.shared.b16 "
                 "{%0, %1, %2, %3}, [%4], 1, %5, %6;\n"
                 : "=r"(value[0]), "=r"(value[2]), "=r"(value[1]),
                   "=r"(value[3])
                 : "r"(smem_int_ptr), "r"(64 - mode * 32), "r"(mode));
  }
}

TL_DEVICE void tix_ldmatrix_x1_trans(void const *const smem_ptr,
                                     void *const local_ptr) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  int32_t *value = reinterpret_cast<int32_t *>(local_ptr);
#if defined(__HGGC_ARCH__)
  asm volatile(
#if __HGGC_ARCH__ == 100
      "ppu.tc01.ex.ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"
#elif __HGGC_ARCH__ == 150
      "ppu.tc02.ldmatrix.sync.aligned.x1.trans.m8n8.shared.b16 {%0}, [%1];\n"
#endif
      : "=r"(value[0])
      : "r"(smem_int_ptr));
#endif
}

TL_DEVICE void tix_ldmatrix_x2_trans(void const *const smem_ptr,
                                     void *const local_ptr) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  int32_t *value = reinterpret_cast<int32_t *>(local_ptr);
#if defined(__HGGC_ARCH__)
  asm volatile(
#if __HGGC_ARCH__ == 100
      "ppu.tc01.ex.ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, "
      "[%2];\n"
#elif __HGGC_ARCH__ == 150
      "ppu.tc02.ldmatrix.sync.aligned.x2.trans.m8n8.shared.b16 {%0, %1}, "
      "[%2];\n"
#endif
      : "=r"(value[0]), "=r"(value[1])
      : "r"(smem_int_ptr));
#endif
}

TL_DEVICE void tix_ldmatrix_x4_trans(void const *const smem_ptr,
                                     void *const local_ptr) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  int32_t *value = reinterpret_cast<int32_t *>(local_ptr);
#if defined(__HGGC_ARCH__)
  asm volatile(
#if __HGGC_ARCH__ == 100
      "ppu.tc01.ex.ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, "
      "%3}, [%4];\n"
#elif __HGGC_ARCH__ == 150
      "ppu.tc02.ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, "
      "%3}, [%4];\n"
#endif
      : "=r"(value[0]), "=r"(value[1]), "=r"(value[2]), "=r"(value[3])
      : "r"(smem_int_ptr));
#endif
}

TL_DEVICE void tix_ldmatrix_x4_trans_ppu(void const *const smem_ptr,
                                         void *const local_ptr) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  int32_t *value = reinterpret_cast<int32_t *>(local_ptr);
#if defined(__HGGC_ARCH__)
  asm volatile(
#if __HGGC_ARCH__ == 100
      "ppu.tc01.ldmatrix.sync.aligned.x1.trans.m16n16.shared.b16 {%0, %1, %2, "
      "%3}, [%4];\n"
#elif __HGGC_ARCH__ == 150
      "ppu.tc02.ldmatrix.sync.aligned.x4.trans.m8n8.shared.b16 {%0, %1, %2, "
      "%3}, [%4];\n"
#endif
      : "=r"(value[0]), "=r"(value[1]), "=r"(value[2]), "=r"(value[3])
      : "r"(smem_int_ptr));
#endif
}

TL_DEVICE void tix_ldmatrix_swzl_x4_trans(void const *const smem_ptr,
                                          void *const local_ptr, uint32_t mode,
                                          bool trans_block) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr) / 16;
  int32_t *value = reinterpret_cast<int32_t *>(local_ptr);
  if (!trans_block) {
    asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.trans.shared."
                 "b16 {%0, %1, %2, %3}, [%4], 1, %5, %6;\n"
                 : "=r"(value[0]), "=r"(value[1]), "=r"(value[2]),
                   "=r"(value[3])
                 : "r"(smem_int_ptr), "r"(64 - mode * 32), "r"(mode));
  } else {
    asm volatile("ppu.tc02.ldmatrix.swzl.sync.bulk.tensor.m8n8.x4.trans.shared."
                 "b16 {%0, %1, %2, %3}, [%4], 1, %5, %6;\n"
                 : "=r"(value[0]), "=r"(value[2]), "=r"(value[1]),
                   "=r"(value[3])
                 : "r"(smem_int_ptr), "r"(64 - mode * 32), "r"(mode));
  }
}

// PPU0015: 8-bit transposed loads at m16n16 granularity.
// hgcc inline-ptx only accepts the plain PTX spelling; the ppu.tc02.* prefixed
// form of m16n16.b8 is not a valid instruction on this arch (verified against
// the compiler's inline-ptx diagnostics: "unsupported ldmatrix in arch
// ppu0015").
TL_DEVICE void tix_ldmatrix_x2_trans_m16n16_b8(void const *const smem_ptr,
                                               void *const local_ptr,
                                               bool trans_block) {
  uint32_t smem_int_ptr = smem_ptr_to_uint(smem_ptr);
  int32_t *value = reinterpret_cast<int32_t *>(local_ptr);
#if defined(__HGGC_ARCH__)
  if (!trans_block) {
    asm volatile("ldmatrix.sync.aligned.x2.trans.m16n16.shared.b8 {%0, %1, %2, "
                 "%3}, [%4];\n"
                 : "=r"(value[0]), "=r"(value[1]), "=r"(value[2]),
                   "=r"(value[3])
                 : "r"(smem_int_ptr));
  } else {
    asm volatile("ldmatrix.sync.aligned.x2.trans.m16n16.shared.b8 {%0, %1, %2, "
                 "%3}, [%4];\n"
                 : "=r"(value[0]), "=r"(value[2]), "=r"(value[1]),
                   "=r"(value[3])
                 : "r"(smem_int_ptr));
  }
#endif
}

} // namespace tl
