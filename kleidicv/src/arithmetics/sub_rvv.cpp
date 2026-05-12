// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>

#include "kleidicv/kleidicv.h"

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

template <typename T>
kleidicv_error_t saturating_sub(const T *src_a, size_t src_a_stride,
                                const T *src_b, size_t src_b_stride, T *dst,
                                size_t dst_stride, size_t width,
                                size_t height) {
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
      auto vr = rvv_ssub(va, vb, vl);
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
      if (std::numeric_limits<T>::is_signed && sb[i] < 0) {
        T result;
        d[i] = __builtin_sub_overflow(sa[i], sb[i], &result)
                   ? std::numeric_limits<T>::max()
                   : result;
      } else {
        T result;
        d[i] = __builtin_sub_overflow(sa[i], sb[i], &result)
                   ? std::numeric_limits<T>::lowest()
                   : result;
      }
    }
  }
#endif

  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                                \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t saturating_sub<type>( \
      const type *src_a, size_t src_a_stride, const type *src_b,           \
      size_t src_b_stride, type *dst, size_t dst_stride, size_t width,     \
      size_t height)

KLEIDICV_INSTANTIATE_TEMPLATE(int8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int32_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint32_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int64_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint64_t);

}  // namespace kleidicv::neon
