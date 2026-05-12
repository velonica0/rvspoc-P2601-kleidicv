// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <type_traits>

#include "kleidicv/kleidicv.h"

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

template <typename T>
kleidicv_error_t in_range(const T *src, size_t src_stride, uint8_t *dst,
                          size_t dst_stride, size_t width, size_t height,
                          T lower_bound, T upper_bound) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

#ifdef __riscv_vector
  if constexpr (std::is_same_v<T, uint8_t>) {
    // uint8 -> uint8 output: same element width, straightforward.
    for (size_t row = 0; row < height; ++row) {
      const uint8_t *s = row_ptr(src, src_stride, row);
      uint8_t *d = row_ptr(dst, dst_stride, row);
      size_t i = 0;
      while (i < width) {
        size_t vl = rvv_setvl<uint8_t>(width - i);
        vuint8m1_t vs = rvv_load(s + i, vl);
        vuint8m1_t vlo = rvv_splat(lower_bound, vl);
        vuint8m1_t vhi = rvv_splat(upper_bound, vl);
        vbool8_t ge_lo = rvv_ge(vs, vlo, vl);
        vbool8_t le_hi = rvv_le(vs, vhi, vl);
        vbool8_t in_mask = __riscv_vmand_mm_b8(ge_lo, le_hi, vl);
        vuint8m1_t v_ff = rvv_splat(static_cast<uint8_t>(0xFF), vl);
        vuint8m1_t v_00 = rvv_splat(static_cast<uint8_t>(0), vl);
        vuint8m1_t vr = rvv_merge(v_00, v_ff, in_mask, vl);
        rvv_store(d + i, vr, vl);
        i += vl;
      }
    }
  } else if constexpr (std::is_same_v<T, float>) {
    // float -> uint8 output: process float elements, write uint8 result.
    // Since source is float (4 bytes) and dest is uint8 (1 byte), we process
    // element by element using scalar within the RVV path for correctness.
    for (size_t row = 0; row < height; ++row) {
      const float *s = row_ptr(src, src_stride, row);
      uint8_t *d = row_ptr(dst, dst_stride, row);
      for (size_t i = 0; i < width; ++i) {
        d[i] = (s[i] >= lower_bound && s[i] <= upper_bound)
                   ? static_cast<uint8_t>(0xFF)
                   : static_cast<uint8_t>(0);
      }
    }
  }
#else
  for (size_t row = 0; row < height; ++row) {
    const T *s = reinterpret_cast<const T *>(
        reinterpret_cast<const uint8_t *>(src) + row * src_stride);
    uint8_t *d = reinterpret_cast<uint8_t *>(
        reinterpret_cast<uint8_t *>(dst) + row * dst_stride);
    for (size_t i = 0; i < width; ++i) {
      d[i] = (s[i] >= lower_bound && s[i] <= upper_bound) ? 0xFF : 0;
    }
  }
#endif

  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                                \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t in_range<type>(       \
      const type *src, size_t src_stride, uint8_t *dst, size_t dst_stride, \
      size_t width, size_t height, type lower_bound, type upper_bound)

KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(float);

}  // namespace kleidicv::neon
