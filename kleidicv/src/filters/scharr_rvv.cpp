// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

// Scharr interleaved: computes Scharr derivatives (dx, dy interleaved).
//
// The applied weights for the horizontal approximation (dx):
//      [  -3   0   3 ]   [  3 ]
//  F = [ -10   0  10 ] = [ 10 ] * [ -1, 0, 1 ]
//      [  -3   0   3 ]   [  3 ]
//
// The applied weights for the vertical approximation (dy):
//      [ -3 -10  -3 ]   [ -1 ]
//  F = [  0   0   0 ] = [  0 ] * [ 3, 10, 3 ]
//      [  3  10   3 ]   [  1 ]
//
// The output is a reduced image: (src_width - 2) columns, (src_height - 2) rows.
// Each output element is a pair [dx, dy] stored interleaved.
// The caller passes y_begin=0, y_end=src_height-2 for the full image.
// Output row y is computed from source rows y, y+1, y+2.
// Output column ox maps to source column (ox + src_channels) with neighbors
// at +/- src_channels.

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t kleidicv_scharr_interleaved_stripe_s16_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t src_channels, int16_t *dst, size_t dst_stride, size_t y_begin,
    size_t y_end) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_IMAGE_SIZE(src_width, src_height);

  const size_t out_width = (src_width - 2) * src_channels;
  const size_t ch = src_channels;

  for (size_t y = y_begin; y < y_end; ++y) {
    // Source rows: y, y+1, y+2 (3 rows for the vertical 3x3 kernel).
    const uint8_t *row0 = row_ptr(src, src_stride, y);
    const uint8_t *row1 = row_ptr(src, src_stride, y + 1);
    const uint8_t *row2 = row_ptr(src, src_stride, y + 2);
    int16_t *dst_row = row_ptr(dst, dst_stride, y);

#ifdef __riscv_vector
    size_t ox = 0;
    while (ox < out_width) {
      size_t vl = __riscv_vsetvl_e8mf2(out_width - ox);
      const size_t sx = ox + ch;

      // Load 8 neighbor values from 3 rows as u8, using mf2 (half-LMUL)
      // so that widening to u16m1/i16m1 stays at LMUL=1.
      vuint8mf2_t r0l = __riscv_vle8_v_u8mf2(row0 + sx - ch, vl);
      vuint8mf2_t r0c = __riscv_vle8_v_u8mf2(row0 + sx, vl);
      vuint8mf2_t r0r = __riscv_vle8_v_u8mf2(row0 + sx + ch, vl);
      vuint8mf2_t r1l = __riscv_vle8_v_u8mf2(row1 + sx - ch, vl);
      vuint8mf2_t r1r = __riscv_vle8_v_u8mf2(row1 + sx + ch, vl);
      vuint8mf2_t r2l = __riscv_vle8_v_u8mf2(row2 + sx - ch, vl);
      vuint8mf2_t r2c = __riscv_vle8_v_u8mf2(row2 + sx, vl);
      vuint8mf2_t r2r = __riscv_vle8_v_u8mf2(row2 + sx + ch, vl);

      // Widen u8 -> u16 -> reinterpret as i16 for signed arithmetic.
      // All source values are [0, 255], so the u16 values are non-negative
      // and safe to reinterpret as i16 (max value 255 * 16 = 4080 < 32767).
      vint16m1_t p0l = __riscv_vreinterpret_v_u16m1_i16m1(
          __riscv_vwcvtu_x_x_v_u16m1(r0l, vl));
      vint16m1_t p0c = __riscv_vreinterpret_v_u16m1_i16m1(
          __riscv_vwcvtu_x_x_v_u16m1(r0c, vl));
      vint16m1_t p0r = __riscv_vreinterpret_v_u16m1_i16m1(
          __riscv_vwcvtu_x_x_v_u16m1(r0r, vl));
      vint16m1_t p1l = __riscv_vreinterpret_v_u16m1_i16m1(
          __riscv_vwcvtu_x_x_v_u16m1(r1l, vl));
      vint16m1_t p1r = __riscv_vreinterpret_v_u16m1_i16m1(
          __riscv_vwcvtu_x_x_v_u16m1(r1r, vl));
      vint16m1_t p2l = __riscv_vreinterpret_v_u16m1_i16m1(
          __riscv_vwcvtu_x_x_v_u16m1(r2l, vl));
      vint16m1_t p2c = __riscv_vreinterpret_v_u16m1_i16m1(
          __riscv_vwcvtu_x_x_v_u16m1(r2c, vl));
      vint16m1_t p2r = __riscv_vreinterpret_v_u16m1_i16m1(
          __riscv_vwcvtu_x_x_v_u16m1(r2r, vl));

      // dx = -3*(p0l - p0r) - 10*(p1l - p1r) - 3*(p2l - p2r)
      //    = 3*(p0r - p0l) + 10*(p1r - p1l) + 3*(p2r - p2l)
      //
      // Compute (right - left) differences first:
      vint16m1_t d0 = __riscv_vsub_vv_i16m1(p0r, p0l, vl);  // row0: right-left
      vint16m1_t d1 = __riscv_vsub_vv_i16m1(p1r, p1l, vl);  // row1: right-left
      vint16m1_t d2 = __riscv_vsub_vv_i16m1(p2r, p2l, vl);  // row2: right-left

      // dx = 3*d0 + 10*d1 + 3*d2
      vint16m1_t vdx = __riscv_vmul_vx_i16m1(d0, 3, vl);
      vdx = __riscv_vmacc_vx_i16m1(vdx, 10, d1, vl);
      vdx = __riscv_vmacc_vx_i16m1(vdx, 3, d2, vl);

      // dy = -3*(p0l + p0r) - 10*p0c + 3*(p2l + p2r) + 10*p2c
      //    = 3*(p2l - p0l) + 10*(p2c - p0c) + 3*(p2r - p0r)
      //
      // Compute (bottom - top) differences:
      vint16m1_t v0 = __riscv_vsub_vv_i16m1(p2l, p0l, vl);  // col-1: bot-top
      vint16m1_t v1 = __riscv_vsub_vv_i16m1(p2c, p0c, vl);  // col 0: bot-top
      vint16m1_t v2 = __riscv_vsub_vv_i16m1(p2r, p0r, vl);  // col+1: bot-top

      // dy = 3*v0 + 10*v1 + 3*v2
      vint16m1_t vdy = __riscv_vmul_vx_i16m1(v0, 3, vl);
      vdy = __riscv_vmacc_vx_i16m1(vdy, 10, v1, vl);
      vdy = __riscv_vmacc_vx_i16m1(vdy, 3, v2, vl);

      // Store interleaved [dx, dy] pairs using segment store.
      vint16m1x2_t out = __riscv_vcreate_v_i16m1x2(vdx, vdy);
      __riscv_vsseg2e16_v_i16m1x2(dst_row + ox * 2, out, vl);

      ox += vl;
    }
#else
    // Scalar fallback.
    for (size_t ox = 0; ox < out_width; ++ox) {
      // Map output position to source position (skip first column).
      const size_t sx = ox + ch;

      int p0l = static_cast<int>(row0[sx - ch]);
      int p0c = static_cast<int>(row0[sx]);
      int p0r = static_cast<int>(row0[sx + ch]);
      int p1l = static_cast<int>(row1[sx - ch]);
      int p1r = static_cast<int>(row1[sx + ch]);
      int p2l = static_cast<int>(row2[sx - ch]);
      int p2c = static_cast<int>(row2[sx]);
      int p2r = static_cast<int>(row2[sx + ch]);

      int dx = -3 * p0l + 3 * p0r - 10 * p1l + 10 * p1r - 3 * p2l + 3 * p2r;
      int dy = -3 * p0l - 10 * p0c - 3 * p0r + 3 * p2l + 10 * p2c + 3 * p2r;

      dst_row[ox * 2] = static_cast<int16_t>(dx);
      dst_row[ox * 2 + 1] = static_cast<int16_t>(dy);
    }
#endif
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::neon
