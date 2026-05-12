// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

template <typename T, typename TInternal>
kleidicv_error_t sum(const T *src, size_t src_stride, size_t width,
                     size_t height, T *sum) {
  CHECK_POINTERS(sum);
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_IMAGE_SIZE(width, height);

#ifdef __riscv_vector
  // RVV vectorized sum using widening to double (float->double not available
  // in RVV m1 easily, use scalar accumulation in double).
  double acc = 0.0;
  for (size_t r = 0; r < height; ++r) {
    const T *row = row_ptr(src, src_stride, r);
    size_t remaining = width;
    size_t i = 0;
    while (remaining > 0) {
      size_t vl = rvv_setvl_e32(remaining);
      vfloat32m1_t v = rvv_load(row + i, vl);
      acc += static_cast<double>(rvv_reduce_sum_f32(v, vl));
      i += vl;
      remaining -= vl;
    }
  }
  *sum = static_cast<T>(acc);
#else
  // Scalar fallback
  TInternal acc = 0;
  for (size_t r = 0; r < height; ++r) {
    const T *row = row_ptr(src, src_stride, r);
    for (size_t c = 0; c < width; ++c) {
      acc += static_cast<TInternal>(row[c]);
    }
  }
  *sum = static_cast<T>(acc);
#endif

  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type, type_internal)                     \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t sum<type, type_internal>( \
      const type *src, size_t src_stride, size_t width, size_t height,         \
      type *sum)

KLEIDICV_INSTANTIATE_TEMPLATE(float, double);

}  // namespace kleidicv::neon
