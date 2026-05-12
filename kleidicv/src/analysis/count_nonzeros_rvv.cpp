// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>
#include <type_traits>

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

template <typename T>
KLEIDICV_TARGET_FN_ATTRS static kleidicv_error_t count_nonzeros(
    const T *src, size_t src_stride, size_t width, size_t height,
    size_t *count) {
  CHECK_POINTERS(count);
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  size_t accumulator = 0;

#ifdef __riscv_vector
  for (size_t r = 0; r < height; ++r) {
    const T *row = row_ptr(src, src_stride, r);
    size_t remaining = width;
    size_t i = 0;
    while (remaining > 0) {
      size_t vl = rvv_setvl_e8(remaining);
      vuint8m1_t v = rvv_load(row + i, vl);
      vuint8m1_t zero = rvv_splat(static_cast<uint8_t>(0), vl);
      // Compare not-equal to zero: mask of non-zero elements
      vbool8_t nz_mask = __riscv_vmsne_vv_u8m1_b8(v, zero, vl);
      accumulator += __riscv_vcpop_m_b8(nz_mask, vl);
      i += vl;
      remaining -= vl;
    }
  }
#else
  // Scalar fallback
  for (size_t r = 0; r < height; ++r) {
    const T *row = row_ptr(src, src_stride, r);
    for (size_t c = 0; c < width; ++c) {
      accumulator += (row[c] != 0) ? 1 : 0;
    }
  }
#endif

  *count = accumulator;
  return KLEIDICV_OK;
}

}  // namespace kleidicv::neon

extern "C" {

decltype(kleidicv::neon::count_nonzeros<uint8_t>) *kleidicv_count_nonzeros_u8 =
    kleidicv::neon::count_nonzeros<uint8_t>;

}  // extern "C"
