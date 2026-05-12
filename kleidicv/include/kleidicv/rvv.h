// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#ifndef KLEIDICV_RVV_H
#define KLEIDICV_RVV_H

#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#include <limits>
#include <type_traits>

#include "kleidicv/config.h"
#include "kleidicv/ctypes.h"
#include "kleidicv/traits.h"
#include "kleidicv/types.h"
#include "kleidicv/utils.h"

#ifdef __riscv_vector
#include <riscv_vector.h>
#endif

namespace kleidicv::neon {

#ifdef __riscv_vector

// --- RVV vsetvl wrappers ---

static inline size_t rvv_setvl_e8(size_t avl) {
  return __riscv_vsetvl_e8m1(avl);
}

static inline size_t rvv_setvl_e16(size_t avl) {
  return __riscv_vsetvl_e16m1(avl);
}

static inline size_t rvv_setvl_e32(size_t avl) {
  return __riscv_vsetvl_e32m1(avl);
}

static inline size_t rvv_setvl_e64(size_t avl) {
  return __riscv_vsetvl_e64m1(avl);
}

template <typename T>
static inline size_t rvv_setvl(size_t avl) {
  if constexpr (sizeof(T) == 1)
    return rvv_setvl_e8(avl);
  else if constexpr (sizeof(T) == 2)
    return rvv_setvl_e16(avl);
  else if constexpr (sizeof(T) == 4)
    return rvv_setvl_e32(avl);
  else
    return rvv_setvl_e64(avl);
}

// --- RVV load wrappers ---

static inline vint8m1_t rvv_load(const int8_t *src, size_t vl) {
  return __riscv_vle8_v_i8m1(src, vl);
}
static inline vuint8m1_t rvv_load(const uint8_t *src, size_t vl) {
  return __riscv_vle8_v_u8m1(src, vl);
}
static inline vint16m1_t rvv_load(const int16_t *src, size_t vl) {
  return __riscv_vle16_v_i16m1(src, vl);
}
static inline vuint16m1_t rvv_load(const uint16_t *src, size_t vl) {
  return __riscv_vle16_v_u16m1(src, vl);
}
static inline vint32m1_t rvv_load(const int32_t *src, size_t vl) {
  return __riscv_vle32_v_i32m1(src, vl);
}
static inline vuint32m1_t rvv_load(const uint32_t *src, size_t vl) {
  return __riscv_vle32_v_u32m1(src, vl);
}
static inline vint64m1_t rvv_load(const int64_t *src, size_t vl) {
  return __riscv_vle64_v_i64m1(src, vl);
}
static inline vuint64m1_t rvv_load(const uint64_t *src, size_t vl) {
  return __riscv_vle64_v_u64m1(src, vl);
}
static inline vfloat32m1_t rvv_load(const float *src, size_t vl) {
  return __riscv_vle32_v_f32m1(src, vl);
}

// --- RVV store wrappers ---

static inline void rvv_store(int8_t *dst, vint8m1_t v, size_t vl) {
  __riscv_vse8_v_i8m1(dst, v, vl);
}
static inline void rvv_store(uint8_t *dst, vuint8m1_t v, size_t vl) {
  __riscv_vse8_v_u8m1(dst, v, vl);
}
static inline void rvv_store(int16_t *dst, vint16m1_t v, size_t vl) {
  __riscv_vse16_v_i16m1(dst, v, vl);
}
static inline void rvv_store(uint16_t *dst, vuint16m1_t v, size_t vl) {
  __riscv_vse16_v_u16m1(dst, v, vl);
}
static inline void rvv_store(int32_t *dst, vint32m1_t v, size_t vl) {
  __riscv_vse32_v_i32m1(dst, v, vl);
}
static inline void rvv_store(uint32_t *dst, vuint32m1_t v, size_t vl) {
  __riscv_vse32_v_u32m1(dst, v, vl);
}
static inline void rvv_store(int64_t *dst, vint64m1_t v, size_t vl) {
  __riscv_vse64_v_i64m1(dst, v, vl);
}
static inline void rvv_store(uint64_t *dst, vuint64m1_t v, size_t vl) {
  __riscv_vse64_v_u64m1(dst, v, vl);
}
static inline void rvv_store(float *dst, vfloat32m1_t v, size_t vl) {
  __riscv_vse32_v_f32m1(dst, v, vl);
}

// --- RVV saturating add ---

static inline vint8m1_t rvv_sadd(vint8m1_t a, vint8m1_t b, size_t vl) {
  return __riscv_vsadd_vv_i8m1(a, b, vl);
}
static inline vuint8m1_t rvv_sadd(vuint8m1_t a, vuint8m1_t b, size_t vl) {
  return __riscv_vsaddu_vv_u8m1(a, b, vl);
}
static inline vint16m1_t rvv_sadd(vint16m1_t a, vint16m1_t b, size_t vl) {
  return __riscv_vsadd_vv_i16m1(a, b, vl);
}
static inline vuint16m1_t rvv_sadd(vuint16m1_t a, vuint16m1_t b, size_t vl) {
  return __riscv_vsaddu_vv_u16m1(a, b, vl);
}
static inline vint32m1_t rvv_sadd(vint32m1_t a, vint32m1_t b, size_t vl) {
  return __riscv_vsadd_vv_i32m1(a, b, vl);
}
static inline vuint32m1_t rvv_sadd(vuint32m1_t a, vuint32m1_t b, size_t vl) {
  return __riscv_vsaddu_vv_u32m1(a, b, vl);
}
static inline vint64m1_t rvv_sadd(vint64m1_t a, vint64m1_t b, size_t vl) {
  return __riscv_vsadd_vv_i64m1(a, b, vl);
}
static inline vuint64m1_t rvv_sadd(vuint64m1_t a, vuint64m1_t b, size_t vl) {
  return __riscv_vsaddu_vv_u64m1(a, b, vl);
}

// --- RVV saturating sub ---

static inline vint8m1_t rvv_ssub(vint8m1_t a, vint8m1_t b, size_t vl) {
  return __riscv_vssub_vv_i8m1(a, b, vl);
}
static inline vuint8m1_t rvv_ssub(vuint8m1_t a, vuint8m1_t b, size_t vl) {
  return __riscv_vssubu_vv_u8m1(a, b, vl);
}
static inline vint16m1_t rvv_ssub(vint16m1_t a, vint16m1_t b, size_t vl) {
  return __riscv_vssub_vv_i16m1(a, b, vl);
}
static inline vuint16m1_t rvv_ssub(vuint16m1_t a, vuint16m1_t b, size_t vl) {
  return __riscv_vssubu_vv_u16m1(a, b, vl);
}
static inline vint32m1_t rvv_ssub(vint32m1_t a, vint32m1_t b, size_t vl) {
  return __riscv_vssub_vv_i32m1(a, b, vl);
}
static inline vuint32m1_t rvv_ssub(vuint32m1_t a, vuint32m1_t b, size_t vl) {
  return __riscv_vssubu_vv_u32m1(a, b, vl);
}
static inline vint64m1_t rvv_ssub(vint64m1_t a, vint64m1_t b, size_t vl) {
  return __riscv_vssub_vv_i64m1(a, b, vl);
}
static inline vuint64m1_t rvv_ssub(vuint64m1_t a, vuint64m1_t b, size_t vl) {
  return __riscv_vssubu_vv_u64m1(a, b, vl);
}

// --- RVV add (non-saturating) ---

static inline vint8m1_t rvv_add(vint8m1_t a, vint8m1_t b, size_t vl) {
  return __riscv_vadd_vv_i8m1(a, b, vl);
}
static inline vuint8m1_t rvv_add(vuint8m1_t a, vuint8m1_t b, size_t vl) {
  return __riscv_vadd_vv_u8m1(a, b, vl);
}
static inline vint16m1_t rvv_add(vint16m1_t a, vint16m1_t b, size_t vl) {
  return __riscv_vadd_vv_i16m1(a, b, vl);
}
static inline vuint16m1_t rvv_add(vuint16m1_t a, vuint16m1_t b, size_t vl) {
  return __riscv_vadd_vv_u16m1(a, b, vl);
}
static inline vint32m1_t rvv_add(vint32m1_t a, vint32m1_t b, size_t vl) {
  return __riscv_vadd_vv_i32m1(a, b, vl);
}
static inline vuint32m1_t rvv_add(vuint32m1_t a, vuint32m1_t b, size_t vl) {
  return __riscv_vadd_vv_u32m1(a, b, vl);
}
static inline vfloat32m1_t rvv_add(vfloat32m1_t a, vfloat32m1_t b, size_t vl) {
  return __riscv_vfadd_vv_f32m1(a, b, vl);
}

// --- RVV sub (non-saturating) ---

static inline vint8m1_t rvv_sub(vint8m1_t a, vint8m1_t b, size_t vl) {
  return __riscv_vsub_vv_i8m1(a, b, vl);
}
static inline vuint8m1_t rvv_sub(vuint8m1_t a, vuint8m1_t b, size_t vl) {
  return __riscv_vsub_vv_u8m1(a, b, vl);
}
static inline vint16m1_t rvv_sub(vint16m1_t a, vint16m1_t b, size_t vl) {
  return __riscv_vsub_vv_i16m1(a, b, vl);
}
static inline vuint16m1_t rvv_sub(vuint16m1_t a, vuint16m1_t b, size_t vl) {
  return __riscv_vsub_vv_u16m1(a, b, vl);
}
static inline vint32m1_t rvv_sub(vint32m1_t a, vint32m1_t b, size_t vl) {
  return __riscv_vsub_vv_i32m1(a, b, vl);
}
static inline vuint32m1_t rvv_sub(vuint32m1_t a, vuint32m1_t b, size_t vl) {
  return __riscv_vsub_vv_u32m1(a, b, vl);
}

// --- RVV bitwise AND ---

static inline vuint8m1_t rvv_and(vuint8m1_t a, vuint8m1_t b, size_t vl) {
  return __riscv_vand_vv_u8m1(a, b, vl);
}
static inline vuint16m1_t rvv_and(vuint16m1_t a, vuint16m1_t b, size_t vl) {
  return __riscv_vand_vv_u16m1(a, b, vl);
}
static inline vuint32m1_t rvv_and(vuint32m1_t a, vuint32m1_t b, size_t vl) {
  return __riscv_vand_vv_u32m1(a, b, vl);
}
static inline vuint64m1_t rvv_and(vuint64m1_t a, vuint64m1_t b, size_t vl) {
  return __riscv_vand_vv_u64m1(a, b, vl);
}

// --- RVV min/max ---

static inline vint8m1_t rvv_min(vint8m1_t a, vint8m1_t b, size_t vl) {
  return __riscv_vmin_vv_i8m1(a, b, vl);
}
static inline vuint8m1_t rvv_min(vuint8m1_t a, vuint8m1_t b, size_t vl) {
  return __riscv_vminu_vv_u8m1(a, b, vl);
}
static inline vint16m1_t rvv_min(vint16m1_t a, vint16m1_t b, size_t vl) {
  return __riscv_vmin_vv_i16m1(a, b, vl);
}
static inline vuint16m1_t rvv_min(vuint16m1_t a, vuint16m1_t b, size_t vl) {
  return __riscv_vminu_vv_u16m1(a, b, vl);
}
static inline vint32m1_t rvv_min(vint32m1_t a, vint32m1_t b, size_t vl) {
  return __riscv_vmin_vv_i32m1(a, b, vl);
}
static inline vuint32m1_t rvv_min(vuint32m1_t a, vuint32m1_t b, size_t vl) {
  return __riscv_vminu_vv_u32m1(a, b, vl);
}
static inline vfloat32m1_t rvv_min(vfloat32m1_t a, vfloat32m1_t b, size_t vl) {
  return __riscv_vfmin_vv_f32m1(a, b, vl);
}

static inline vint8m1_t rvv_max(vint8m1_t a, vint8m1_t b, size_t vl) {
  return __riscv_vmax_vv_i8m1(a, b, vl);
}
static inline vuint8m1_t rvv_max(vuint8m1_t a, vuint8m1_t b, size_t vl) {
  return __riscv_vmaxu_vv_u8m1(a, b, vl);
}
static inline vint16m1_t rvv_max(vint16m1_t a, vint16m1_t b, size_t vl) {
  return __riscv_vmax_vv_i16m1(a, b, vl);
}
static inline vuint16m1_t rvv_max(vuint16m1_t a, vuint16m1_t b, size_t vl) {
  return __riscv_vmaxu_vv_u16m1(a, b, vl);
}
static inline vint32m1_t rvv_max(vint32m1_t a, vint32m1_t b, size_t vl) {
  return __riscv_vmax_vv_i32m1(a, b, vl);
}
static inline vuint32m1_t rvv_max(vuint32m1_t a, vuint32m1_t b, size_t vl) {
  return __riscv_vmaxu_vv_u32m1(a, b, vl);
}
static inline vfloat32m1_t rvv_max(vfloat32m1_t a, vfloat32m1_t b, size_t vl) {
  return __riscv_vfmax_vv_f32m1(a, b, vl);
}

// --- RVV broadcast/splat ---

static inline vint8m1_t rvv_splat(int8_t val, size_t vl) {
  return __riscv_vmv_v_x_i8m1(val, vl);
}
static inline vuint8m1_t rvv_splat(uint8_t val, size_t vl) {
  return __riscv_vmv_v_x_u8m1(val, vl);
}
static inline vint16m1_t rvv_splat(int16_t val, size_t vl) {
  return __riscv_vmv_v_x_i16m1(val, vl);
}
static inline vuint16m1_t rvv_splat(uint16_t val, size_t vl) {
  return __riscv_vmv_v_x_u16m1(val, vl);
}
static inline vint32m1_t rvv_splat(int32_t val, size_t vl) {
  return __riscv_vmv_v_x_i32m1(val, vl);
}
static inline vuint32m1_t rvv_splat(uint32_t val, size_t vl) {
  return __riscv_vmv_v_x_u32m1(val, vl);
}
static inline vint64m1_t rvv_splat(int64_t val, size_t vl) {
  return __riscv_vmv_v_x_i64m1(val, vl);
}
static inline vuint64m1_t rvv_splat(uint64_t val, size_t vl) {
  return __riscv_vmv_v_x_u64m1(val, vl);
}
static inline vfloat32m1_t rvv_splat_f32(float val, size_t vl) {
  return __riscv_vfmv_v_f_f32m1(val, vl);
}

// --- RVV multiply ---

static inline vuint8m1_t rvv_mul(vuint8m1_t a, vuint8m1_t b, size_t vl) {
  return __riscv_vmul_vv_u8m1(a, b, vl);
}
static inline vint16m1_t rvv_mul(vint16m1_t a, vint16m1_t b, size_t vl) {
  return __riscv_vmul_vv_i16m1(a, b, vl);
}
static inline vuint16m1_t rvv_mul(vuint16m1_t a, vuint16m1_t b, size_t vl) {
  return __riscv_vmul_vv_u16m1(a, b, vl);
}
static inline vint32m1_t rvv_mul(vint32m1_t a, vint32m1_t b, size_t vl) {
  return __riscv_vmul_vv_i32m1(a, b, vl);
}
static inline vuint32m1_t rvv_mul(vuint32m1_t a, vuint32m1_t b, size_t vl) {
  return __riscv_vmul_vv_u32m1(a, b, vl);
}
static inline vfloat32m1_t rvv_mul(vfloat32m1_t a, vfloat32m1_t b, size_t vl) {
  return __riscv_vfmul_vv_f32m1(a, b, vl);
}

// --- RVV widening operations ---

static inline vuint16m2_t rvv_wmulu(vuint8m1_t a, vuint8m1_t b, size_t vl) {
  return __riscv_vwmulu_vv_u16m2(a, b, vl);
}
static inline vint16m2_t rvv_wmul(vint8m1_t a, vint8m1_t b, size_t vl) {
  return __riscv_vwmul_vv_i16m2(a, b, vl);
}

// --- RVV comparison (return mask) ---

static inline vbool8_t rvv_eq(vuint8m1_t a, vuint8m1_t b, size_t vl) {
  return __riscv_vmseq_vv_u8m1_b8(a, b, vl);
}
static inline vbool8_t rvv_gt(vuint8m1_t a, vuint8m1_t b, size_t vl) {
  return __riscv_vmsgtu_vv_u8m1_b8(a, b, vl);
}
static inline vbool8_t rvv_ge(vuint8m1_t a, vuint8m1_t b, size_t vl) {
  return __riscv_vmsgeu_vv_u8m1_b8(a, b, vl);
}
static inline vbool8_t rvv_le(vuint8m1_t a, vuint8m1_t b, size_t vl) {
  return __riscv_vmsleu_vv_u8m1_b8(a, b, vl);
}

// --- RVV merge based on mask ---

static inline vuint8m1_t rvv_merge(vuint8m1_t a, vuint8m1_t b, vbool8_t mask,
                                   size_t vl) {
  return __riscv_vmerge_vvm_u8m1(a, b, mask, vl);
}

// --- RVV reduction operations ---

static inline uint32_t rvv_reduce_sum_u32(vuint32m1_t v, size_t vl) {
  vuint32m1_t zero = __riscv_vmv_v_x_u32m1(0, vl);
  vuint32m1_t result = __riscv_vredsum_vs_u32m1_u32m1(v, zero, vl);
  return __riscv_vmv_x_s_u32m1_u32(result);
}

static inline int32_t rvv_reduce_sum_i32(vint32m1_t v, size_t vl) {
  vint32m1_t zero = __riscv_vmv_v_x_i32m1(0, vl);
  vint32m1_t result = __riscv_vredsum_vs_i32m1_i32m1(v, zero, vl);
  return __riscv_vmv_x_s_i32m1_i32(result);
}

static inline float rvv_reduce_sum_f32(vfloat32m1_t v, size_t vl) {
  vfloat32m1_t zero = __riscv_vfmv_v_f_f32m1(0.0f, vl);
  vfloat32m1_t result = __riscv_vfredusum_vs_f32m1_f32m1(v, zero, vl);
  return __riscv_vfmv_f_s_f32m1_f32(result);
}

// --- RVV shift operations ---

static inline vuint8m1_t rvv_srl(vuint8m1_t a, size_t shift, size_t vl) {
  return __riscv_vsrl_vx_u8m1(a, shift, vl);
}
static inline vuint16m1_t rvv_srl(vuint16m1_t a, size_t shift, size_t vl) {
  return __riscv_vsrl_vx_u16m1(a, shift, vl);
}
static inline vuint32m1_t rvv_srl(vuint32m1_t a, size_t shift, size_t vl) {
  return __riscv_vsrl_vx_u32m1(a, shift, vl);
}
static inline vint8m1_t rvv_sra(vint8m1_t a, size_t shift, size_t vl) {
  return __riscv_vsra_vx_i8m1(a, shift, vl);
}
static inline vint16m1_t rvv_sra(vint16m1_t a, size_t shift, size_t vl) {
  return __riscv_vsra_vx_i16m1(a, shift, vl);
}
static inline vint32m1_t rvv_sra(vint32m1_t a, size_t shift, size_t vl) {
  return __riscv_vsra_vx_i32m1(a, shift, vl);
}

// --- RVV narrowing operations ---

static inline vuint8m1_t rvv_narrow_u16_to_u8(vuint16m2_t v, size_t vl) {
  return __riscv_vnclipu_wx_u8m1(v, 0, __RISCV_VXRM_RDN, vl);
}
static inline vint8m1_t rvv_narrow_i16_to_i8(vint16m2_t v, size_t vl) {
  return __riscv_vnclip_wx_i8m1(v, 0, __RISCV_VXRM_RDN, vl);
}

// --- RVV absolute difference ---
// |a - b| for unsigned types: max(a,b) - min(a,b)

static inline vuint8m1_t rvv_absdiff(vuint8m1_t a, vuint8m1_t b, size_t vl) {
  vuint8m1_t mx = __riscv_vmaxu_vv_u8m1(a, b, vl);
  vuint8m1_t mn = __riscv_vminu_vv_u8m1(a, b, vl);
  return __riscv_vsub_vv_u8m1(mx, mn, vl);
}
static inline vuint16m1_t rvv_absdiff(vuint16m1_t a, vuint16m1_t b,
                                      size_t vl) {
  vuint16m1_t mx = __riscv_vmaxu_vv_u16m1(a, b, vl);
  vuint16m1_t mn = __riscv_vminu_vv_u16m1(a, b, vl);
  return __riscv_vsub_vv_u16m1(mx, mn, vl);
}
static inline vuint32m1_t rvv_absdiff(vuint32m1_t a, vuint32m1_t b,
                                      size_t vl) {
  vuint32m1_t mx = __riscv_vmaxu_vv_u32m1(a, b, vl);
  vuint32m1_t mn = __riscv_vminu_vv_u32m1(a, b, vl);
  return __riscv_vsub_vv_u32m1(mx, mn, vl);
}

// Signed absdiff via widening to avoid overflow
static inline vint8m1_t rvv_absdiff(vint8m1_t a, vint8m1_t b, size_t vl) {
  vint16m2_t wa = __riscv_vsext_vf2_i16m2(a, vl);
  vint16m2_t wb = __riscv_vsext_vf2_i16m2(b, vl);
  vint16m2_t diff = __riscv_vsub_vv_i16m2(wa, wb, vl);
  vint16m2_t neg = __riscv_vneg_v_i16m2(diff, vl);
  vint16m2_t abs_diff = __riscv_vmax_vv_i16m2(diff, neg, vl);
  return __riscv_vnclip_wx_i8m1(abs_diff, 0, __RISCV_VXRM_RDN, vl);
}
static inline vint16m1_t rvv_absdiff(vint16m1_t a, vint16m1_t b, size_t vl) {
  vint32m2_t wa = __riscv_vsext_vf2_i32m2(a, vl);
  vint32m2_t wb = __riscv_vsext_vf2_i32m2(b, vl);
  vint32m2_t diff = __riscv_vsub_vv_i32m2(wa, wb, vl);
  vint32m2_t neg = __riscv_vneg_v_i32m2(diff, vl);
  vint32m2_t abs_diff = __riscv_vmax_vv_i32m2(diff, neg, vl);
  return __riscv_vnclip_wx_i16m1(abs_diff, 0, __RISCV_VXRM_RDN, vl);
}

#endif  // __riscv_vector

// --- Row-processing helper for RVV kernels ---
// Processes a 2D image row by row, calling the given functor for each row.
// The functor receives (row_src_ptrs..., row_dst_ptr, width).

template <typename T>
static inline const T *row_ptr(const T *base, size_t stride, size_t row) {
  return reinterpret_cast<const T *>(reinterpret_cast<const uint8_t *>(base) +
                                     row * stride);
}

template <typename T>
static inline T *row_ptr(T *base, size_t stride, size_t row) {
  return reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(base) +
                               row * stride);
}

}  // namespace kleidicv::neon

#endif  // KLEIDICV_RVV_H
