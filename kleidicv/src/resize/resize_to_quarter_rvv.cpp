// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

// Quarter-size downsampling: each output pixel is the average of a 2x2 block.
// Output dimensions: src_width/2 x src_height/2.

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t resize_to_quarter_u8(const uint8_t *src, size_t src_stride,
                                      size_t src_width, size_t src_height,
                                      uint8_t *dst, size_t dst_stride) {
  size_t dst_width = src_width / 2;
  size_t dst_height = src_height / 2;

#ifdef __riscv_vector
  for (size_t dy = 0; dy < dst_height; ++dy) {
    const uint8_t *top = row_ptr(src, src_stride, dy * 2);
    const uint8_t *bot = row_ptr(src, src_stride, dy * 2 + 1);
    uint8_t *dst_row = row_ptr(dst, dst_stride, dy);

    size_t dx = 0;
    while (dx < dst_width) {
      size_t vl = __riscv_vsetvl_e8m1(dst_width - dx);
      // Load even and odd pixels from top row
      vuint8m1_t t0 = __riscv_vlse8_v_u8m1(top + dx * 2, 2, vl);     // even
      vuint8m1_t t1 = __riscv_vlse8_v_u8m1(top + dx * 2 + 1, 2, vl); // odd
      // Load even and odd pixels from bottom row
      vuint8m1_t b0 = __riscv_vlse8_v_u8m1(bot + dx * 2, 2, vl);
      vuint8m1_t b1 = __riscv_vlse8_v_u8m1(bot + dx * 2 + 1, 2, vl);

      // Sum all 4 pixels, widen to u16
      vuint16m2_t sum = __riscv_vwaddu_vv_u16m2(t0, t1, vl);
      sum = __riscv_vwaddu_wv_u16m2(sum, b0, vl);
      sum = __riscv_vwaddu_wv_u16m2(sum, b1, vl);

      // Round and divide by 4: (sum + 2) >> 2
      sum = __riscv_vadd_vx_u16m2(sum, 2, vl);
      vuint8m1_t result = __riscv_vnsrl_wx_u8m1(sum, 2, vl);
      __riscv_vse8_v_u8m1(dst_row + dx, result, vl);
      dx += vl;
    }
  }
#else
  // Scalar fallback: 2x2 block averaging.
  for (size_t dy = 0; dy < dst_height; ++dy) {
    const uint8_t *top = row_ptr(src, src_stride, dy * 2);
    const uint8_t *bot = row_ptr(src, src_stride, dy * 2 + 1);
    uint8_t *dst_row = row_ptr(dst, dst_stride, dy);
    for (size_t dx = 0; dx < dst_width; ++dx) {
      uint32_t sum = static_cast<uint32_t>(top[dx * 2]) +
                     static_cast<uint32_t>(top[dx * 2 + 1]) +
                     static_cast<uint32_t>(bot[dx * 2]) +
                     static_cast<uint32_t>(bot[dx * 2 + 1]);
      // Rounding divide by 4
      dst_row[dx] = static_cast<uint8_t>((sum + 2) >> 2);
    }
  }
#endif

  return KLEIDICV_OK;
}

}  // namespace kleidicv::neon
