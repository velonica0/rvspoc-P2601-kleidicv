// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <limits>

#include "kleidicv/kleidicv.h"

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

#ifdef __riscv_vector
// Saturating absolute value for int16 via RVV.
// For the minimum value (-32768), saturate to max (32767).
static inline vint16m1_t rvv_qabs_i16(vint16m1_t v, size_t vl) {
  vint16m1_t neg = __riscv_vneg_v_i16m1(v, vl);
  vint16m1_t abs_v = __riscv_vmax_vv_i16m1(v, neg, vl);
  // Clamp: if original was INT16_MIN, neg overflows to INT16_MIN again.
  // max(INT16_MIN, INT16_MIN) = INT16_MIN, which is still negative.
  // Fix by clamping negatives to INT16_MAX.
  vint16m1_t vmax = __riscv_vmv_v_x_i16m1(std::numeric_limits<int16_t>::max(),
                                           vl);
  vint16m1_t vzero = __riscv_vmv_v_x_i16m1(0, vl);
  vbool16_t neg_mask = __riscv_vmslt_vv_i16m1_b16(abs_v, vzero, vl);
  return __riscv_vmerge_vvm_i16m1(abs_v, vmax, neg_mask, vl);
}
#endif

template <typename T>
kleidicv_error_t saturating_add_abs_with_threshold(
    const T *src_a, size_t src_a_stride, const T *src_b, size_t src_b_stride,
    T *dst, size_t dst_stride, size_t width, size_t height, T threshold) {
  CHECK_POINTER_AND_STRIDE(src_a, src_a_stride, height);
  CHECK_POINTER_AND_STRIDE(src_b, src_b_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

#ifdef __riscv_vector
  for (size_t row = 0; row < height; ++row) {
    const T *sa = row_ptr(src_a, src_a_stride, row);
    const T *sb = row_ptr(src_b, src_b_stride, row);
    T *d = row_ptr(dst, dst_stride, row);
    size_t i = 0;
    while (i < width) {
      size_t vl = rvv_setvl<T>(width - i);
      auto va = rvv_load(sa + i, vl);
      auto vb = rvv_load(sb + i, vl);

      // saturating abs of each input
      auto abs_a = rvv_qabs_i16(va, vl);
      auto abs_b = rvv_qabs_i16(vb, vl);

      // saturating add of the two absolute values
      auto add_abs = rvv_sadd(abs_a, abs_b, vl);

      // threshold: if add_abs > threshold, keep; else zero
      auto vthresh = rvv_splat(threshold, vl);
      auto vzero = rvv_splat(static_cast<T>(0), vl);
      vbool16_t gt_mask = __riscv_vmsgt_vv_i16m1_b16(add_abs, vthresh, vl);
      auto vr = __riscv_vmerge_vvm_i16m1(vzero, add_abs, gt_mask, vl);

      rvv_store(d + i, vr, vl);
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
      // Saturating abs
      auto sat_abs = [](T input) -> T {
        if (std::numeric_limits<T>::is_signed &&
            input == std::numeric_limits<T>::lowest()) {
          return std::numeric_limits<T>::max();
        }
        return std::abs(input);
      };

      T add_abs = 0;
      if (__builtin_add_overflow(sat_abs(sa[i]), sat_abs(sb[i]), &add_abs)) {
        add_abs = std::numeric_limits<T>::max();
      }
      d[i] = add_abs > threshold ? add_abs : 0;
    }
  }
#endif

  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                            \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t                   \
  saturating_add_abs_with_threshold<type>(                             \
      const type *src_a, size_t src_a_stride, const type *src_b,       \
      size_t src_b_stride, type *dst, size_t dst_stride, size_t width, \
      size_t height, type threshold)

KLEIDICV_INSTANTIATE_TEMPLATE(int16_t);

}  // namespace kleidicv::neon
