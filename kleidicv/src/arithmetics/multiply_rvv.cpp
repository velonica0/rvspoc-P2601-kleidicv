// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>

#include "kleidicv/kleidicv.h"
#include "kleidicv/types.h"

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

#ifdef __riscv_vector
// Saturating multiply for uint8: widen to uint16, multiply, narrow-saturate.
static inline void rvv_sat_mul_u8(const uint8_t *sa, const uint8_t *sb,
                                  uint8_t *d, size_t vl) {
  vuint8m1_t va = rvv_load(sa, vl);
  vuint8m1_t vb = rvv_load(sb, vl);
  vuint16m2_t wide = rvv_wmulu(va, vb, vl);
  vuint8m1_t result = rvv_narrow_u16_to_u8(wide, vl);
  rvv_store(d, result, vl);
}

// Saturating multiply for int8: widen to int16, multiply, narrow-saturate.
static inline void rvv_sat_mul_i8(const int8_t *sa, const int8_t *sb,
                                  int8_t *d, size_t vl) {
  vint8m1_t va = rvv_load(sa, vl);
  vint8m1_t vb = rvv_load(sb, vl);
  vint16m2_t wide = rvv_wmul(va, vb, vl);
  vint8m1_t result = rvv_narrow_i16_to_i8(wide, vl);
  rvv_store(d, result, vl);
}

// Saturating multiply for uint16: widen to uint32, multiply, narrow-saturate.
static inline void rvv_sat_mul_u16(const uint16_t *sa, const uint16_t *sb,
                                   uint16_t *d, size_t vl) {
  vuint16m1_t va = rvv_load(sa, vl);
  vuint16m1_t vb = rvv_load(sb, vl);
  vuint32m2_t wide = __riscv_vwmulu_vv_u32m2(va, vb, vl);
  vuint16m1_t result = __riscv_vnclipu_wx_u16m1(wide, 0, __RISCV_VXRM_RDN, vl);
  rvv_store(d, result, vl);
}

// Saturating multiply for int16: widen to int32, multiply, narrow-saturate.
static inline void rvv_sat_mul_i16(const int16_t *sa, const int16_t *sb,
                                   int16_t *d, size_t vl) {
  vint16m1_t va = rvv_load(sa, vl);
  vint16m1_t vb = rvv_load(sb, vl);
  vint32m2_t wide = __riscv_vwmul_vv_i32m2(va, vb, vl);
  vint16m1_t result = __riscv_vnclip_wx_i16m1(wide, 0, __RISCV_VXRM_RDN, vl);
  rvv_store(d, result, vl);
}

// Saturating multiply for int32: widen to int64, multiply, narrow-saturate.
static inline void rvv_sat_mul_i32(const int32_t *sa, const int32_t *sb,
                                   int32_t *d, size_t vl) {
  vint32m1_t va = rvv_load(sa, vl);
  vint32m1_t vb = rvv_load(sb, vl);
  vint64m2_t wide = __riscv_vwmul_vv_i64m2(va, vb, vl);
  vint32m1_t result = __riscv_vnclip_wx_i32m1(wide, 0, __RISCV_VXRM_RDN, vl);
  rvv_store(d, result, vl);
}
#endif

template <typename T>
kleidicv_error_t saturating_multiply(const T *src_a, size_t src_a_stride,
                                     const T *src_b, size_t src_b_stride,
                                     T *dst, size_t dst_stride, size_t width,
                                     size_t height, double scale) {
  CHECK_POINTER_AND_STRIDE(src_a, src_a_stride, height);
  CHECK_POINTER_AND_STRIDE(src_b, src_b_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  (void)scale;  // TODO: figure out the way to process the scale.

#ifdef __riscv_vector
  for (size_t row = 0; row < height; ++row) {
    const T *sa = row_ptr(src_a, src_a_stride, row);
    const T *sb = row_ptr(src_b, src_b_stride, row);
    T *d = row_ptr(dst, dst_stride, row);
    size_t i = 0;
    while (i < width) {
      size_t vl = rvv_setvl<T>(width - i);
      if constexpr (std::is_same_v<T, uint8_t>) {
        rvv_sat_mul_u8(sa + i, sb + i, d + i, vl);
      } else if constexpr (std::is_same_v<T, int8_t>) {
        rvv_sat_mul_i8(sa + i, sb + i, d + i, vl);
      } else if constexpr (std::is_same_v<T, uint16_t>) {
        rvv_sat_mul_u16(sa + i, sb + i, d + i, vl);
      } else if constexpr (std::is_same_v<T, int16_t>) {
        rvv_sat_mul_i16(sa + i, sb + i, d + i, vl);
      } else if constexpr (std::is_same_v<T, int32_t>) {
        rvv_sat_mul_i32(sa + i, sb + i, d + i, vl);
      }
      i += vl;
    }
  }
#else
  for (size_t row = 0; row < height; ++row) {
    const T *sa = reinterpret_cast<const T *>(
        reinterpret_cast<const uint8_t *>(src_a) + row * src_a_stride);
    const T *sb = reinterpret_cast<const T *>(
        reinterpret_cast<const uint8_t *>(src_b) + row * src_b_stride);
    T *d = reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(dst) +
                                 row * dst_stride);
    for (size_t i = 0; i < width; ++i) {
      T result;
      if (std::numeric_limits<T>::is_signed) {
        if (__builtin_mul_overflow(sa[i], sb[i], &result)) {
          d[i] = (sa[i] < 0 && sb[i] > 0) || (sa[i] > 0 && sb[i] < 0)
                     ? std::numeric_limits<T>::lowest()
                     : std::numeric_limits<T>::max();
        } else {
          d[i] = result;
        }
      } else {
        if (__builtin_mul_overflow(sa[i], sb[i], &result)) {
          d[i] = std::numeric_limits<T>::max();
        } else {
          d[i] = result;
        }
      }
    }
  }
#endif

  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                                    \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t                           \
  saturating_multiply<type>(const type *src_a, size_t src_a_stride,            \
                            const type *src_b, size_t src_b_stride, type *dst, \
                            size_t dst_stride, size_t width, size_t height,    \
                            double scale)

KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int32_t);

}  // namespace kleidicv::neon
