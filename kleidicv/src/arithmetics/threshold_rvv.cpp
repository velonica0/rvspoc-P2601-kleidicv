// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/kleidicv.h"

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

template <typename T>
kleidicv_error_t threshold_binary(const T *src, size_t src_stride, T *dst,
                                  size_t dst_stride, size_t width,
                                  size_t height, T threshold, T value) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

#ifdef __riscv_vector
  for (size_t row = 0; row < height; ++row) {
    const T *s = row_ptr(src, src_stride, row);
    T *d = row_ptr(dst, dst_stride, row);
    size_t i = 0;
    while (i < width) {
      size_t vl = rvv_setvl<T>(width - i);
      auto vs = rvv_load(s + i, vl);
      auto vthresh = rvv_splat(threshold, vl);
      auto vval = rvv_splat(value, vl);
      auto vzero = rvv_splat(static_cast<T>(0), vl);
      // mask: src > threshold
      vbool8_t mask = rvv_gt(vs, vthresh, vl);
      auto vr = rvv_merge(vzero, vval, mask, vl);
      rvv_store(d + i, vr, vl);
      i += vl;
    }
  }
#else
  for (size_t row = 0; row < height; ++row) {
    const T *s = reinterpret_cast<const T *>(
        reinterpret_cast<const uint8_t *>(src) + row * src_stride);
    T *d = reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(dst) +
                                 row * dst_stride);
    for (size_t i = 0; i < width; ++i) {
      d[i] = s[i] > threshold ? value : 0;
    }
  }
#endif

  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                                  \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t threshold_binary<type>( \
      const type *src, size_t src_stride, type *dst, size_t dst_stride,      \
      size_t width, size_t height, type threshold, type value)

KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);

}  // namespace kleidicv::neon
