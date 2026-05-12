// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/kleidicv.h"

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

template <typename ScalarType>
kleidicv_error_t compare_equal(const ScalarType *src_a, size_t src_a_stride,
                               const ScalarType *src_b, size_t src_b_stride,
                               ScalarType *dst, size_t dst_stride, size_t width,
                               size_t height) {
  CHECK_POINTER_AND_STRIDE(src_a, src_a_stride, height);
  CHECK_POINTER_AND_STRIDE(src_b, src_b_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

#ifdef __riscv_vector
  for (size_t row = 0; row < height; ++row) {
    const ScalarType *sa = row_ptr(src_a, src_a_stride, row);
    const ScalarType *sb = row_ptr(src_b, src_b_stride, row);
    ScalarType *d = row_ptr(dst, dst_stride, row);
    size_t i = 0;
    while (i < width) {
      size_t vl = rvv_setvl<ScalarType>(width - i);
      auto va = rvv_load(sa + i, vl);
      auto vb = rvv_load(sb + i, vl);
      auto v_ff = rvv_splat(static_cast<ScalarType>(0xFF), vl);
      auto v_00 = rvv_splat(static_cast<ScalarType>(0), vl);
      vbool8_t mask = rvv_eq(va, vb, vl);
      auto vr = rvv_merge(v_00, v_ff, mask, vl);
      rvv_store(d + i, vr, vl);
      i += vl;
    }
  }
#else
  for (size_t row = 0; row < height; ++row) {
    const ScalarType *sa = reinterpret_cast<const ScalarType *>(
        reinterpret_cast<const uint8_t *>(src_a) + row * src_a_stride);
    const ScalarType *sb = reinterpret_cast<const ScalarType *>(
        reinterpret_cast<const uint8_t *>(src_b) + row * src_b_stride);
    ScalarType *d = reinterpret_cast<ScalarType *>(
        reinterpret_cast<uint8_t *>(dst) + row * dst_stride);
    for (size_t i = 0; i < width; ++i) {
      d[i] = sa[i] == sb[i] ? 255 : 0;
    }
  }
#endif

  return KLEIDICV_OK;
}

template <typename ScalarType>
kleidicv_error_t compare_greater(const ScalarType *src_a, size_t src_a_stride,
                                 const ScalarType *src_b, size_t src_b_stride,
                                 ScalarType *dst, size_t dst_stride,
                                 size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src_a, src_a_stride, height);
  CHECK_POINTER_AND_STRIDE(src_b, src_b_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

#ifdef __riscv_vector
  for (size_t row = 0; row < height; ++row) {
    const ScalarType *sa = row_ptr(src_a, src_a_stride, row);
    const ScalarType *sb = row_ptr(src_b, src_b_stride, row);
    ScalarType *d = row_ptr(dst, dst_stride, row);
    size_t i = 0;
    while (i < width) {
      size_t vl = rvv_setvl<ScalarType>(width - i);
      auto va = rvv_load(sa + i, vl);
      auto vb = rvv_load(sb + i, vl);
      auto v_ff = rvv_splat(static_cast<ScalarType>(0xFF), vl);
      auto v_00 = rvv_splat(static_cast<ScalarType>(0), vl);
      vbool8_t mask = rvv_gt(va, vb, vl);
      auto vr = rvv_merge(v_00, v_ff, mask, vl);
      rvv_store(d + i, vr, vl);
      i += vl;
    }
  }
#else
  for (size_t row = 0; row < height; ++row) {
    const ScalarType *sa = reinterpret_cast<const ScalarType *>(
        reinterpret_cast<const uint8_t *>(src_a) + row * src_a_stride);
    const ScalarType *sb = reinterpret_cast<const ScalarType *>(
        reinterpret_cast<const uint8_t *>(src_b) + row * src_b_stride);
    ScalarType *d = reinterpret_cast<ScalarType *>(
        reinterpret_cast<uint8_t *>(dst) + row * dst_stride);
    for (size_t i = 0; i < width; ++i) {
      d[i] = sa[i] > sb[i] ? 255 : 0;
    }
  }
#endif

  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(name, stype)                      \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t name<stype>(       \
      const stype *src_a, size_t src_a_stride, const stype *src_b,      \
      size_t src_b_stride, stype *dst, size_t dst_stride, size_t width, \
      size_t height)

KLEIDICV_INSTANTIATE_TEMPLATE(compare_equal, uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(compare_greater, uint8_t);

}  // namespace kleidicv::neon
