// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

// Helper: scalar Sobel computation for a single pixel (horizontal Gx).
// Kernel: [-1 0 1; -2 0 2; -1 0 1]
static inline int16_t sobel_horiz_scalar(const uint8_t *row_prev,
                                         const uint8_t *row_curr,
                                         const uint8_t *row_next, size_t x,
                                         size_t channels,
                                         size_t total_width) {
  size_t xl = (x >= channels) ? x - channels : x;
  size_t xr = (x + channels < total_width) ? x + channels : x;
  int val = -static_cast<int>(row_prev[xl]) + static_cast<int>(row_prev[xr]) -
            2 * static_cast<int>(row_curr[xl]) +
            2 * static_cast<int>(row_curr[xr]) -
            static_cast<int>(row_next[xl]) + static_cast<int>(row_next[xr]);
  return static_cast<int16_t>(val);
}

// Helper: scalar Sobel computation for a single pixel (vertical Gy).
// Kernel: [-1 -2 -1; 0 0 0; 1 2 1]
static inline int16_t sobel_vert_scalar(const uint8_t *row_prev,
                                        const uint8_t *row_next, size_t x,
                                        size_t channels, size_t total_width) {
  size_t xl = (x >= channels) ? x - channels : x;
  size_t xr = (x + channels < total_width) ? x + channels : x;
  int val = -static_cast<int>(row_prev[xl]) -
            2 * static_cast<int>(row_prev[x]) -
            static_cast<int>(row_prev[xr]) + static_cast<int>(row_next[xl]) +
            2 * static_cast<int>(row_next[x]) +
            static_cast<int>(row_next[xr]);
  return static_cast<int16_t>(val);
}

// Sobel 3x3 horizontal: computes dI/dx using the Sobel operator.
// Kernel: [-1 0 1; -2 0 2; -1 0 1]
KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t sobel_3x3_horizontal_stripe_s16_u8(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end,
    size_t channels) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  size_t total_width = width * channels;

  for (size_t y = y_begin; y < y_end; ++y) {
    const uint8_t *row_prev =
        row_ptr(src, src_stride, (y == 0) ? 0 : y - 1);
    const uint8_t *row_curr = row_ptr(src, src_stride, y);
    const uint8_t *row_next =
        row_ptr(src, src_stride, (y + 1 >= height) ? height - 1 : y + 1);
    int16_t *dst_row = row_ptr(dst, dst_stride, y);

#ifdef __riscv_vector
    if (total_width > 2 * channels) {
      // Left border: scalar for x in [0, channels)
      for (size_t x = 0; x < channels; ++x) {
        dst_row[x] = sobel_horiz_scalar(row_prev, row_curr, row_next, x,
                                        channels, total_width);
      }

      // Interior: RVV for x in [channels, total_width - channels)
      // At position x in this range:
      //   left  load from x - channels  (>= 0, safe)
      //   right load from x + channels  (< total_width when x + channels + vl - 1 < total_width)
      //
      // Since vl <= end - x = total_width - channels - x:
      //   x + channels + vl - 1 <= x + channels + (total_width - channels - x) - 1
      //                         = total_width - 1
      // So the rightmost u8 load is always in bounds.
      {
        size_t x = channels;
        size_t end = total_width - channels;
        while (x < end) {
          size_t vl = __riscv_vsetvl_e16m1(end - x);

          // Load left (x - channels), center (x), right (x + channels)
          // from each row, widen u8 -> i16, compute Sobel Gx
          vuint8mf2_t pl = __riscv_vle8_v_u8mf2(row_prev + x - channels, vl);
          vuint8mf2_t pr = __riscv_vle8_v_u8mf2(row_prev + x + channels, vl);
          vuint8mf2_t cl = __riscv_vle8_v_u8mf2(row_curr + x - channels, vl);
          vuint8mf2_t cr = __riscv_vle8_v_u8mf2(row_curr + x + channels, vl);
          vuint8mf2_t nl = __riscv_vle8_v_u8mf2(row_next + x - channels, vl);
          vuint8mf2_t nr = __riscv_vle8_v_u8mf2(row_next + x + channels, vl);

          // Widen u8 -> u16 (zero-extend)
          vuint16m1_t pl16 = __riscv_vwcvtu_x_x_v_u16m1(pl, vl);
          vuint16m1_t pr16 = __riscv_vwcvtu_x_x_v_u16m1(pr, vl);
          vuint16m1_t cl16 = __riscv_vwcvtu_x_x_v_u16m1(cl, vl);
          vuint16m1_t cr16 = __riscv_vwcvtu_x_x_v_u16m1(cr, vl);
          vuint16m1_t nl16 = __riscv_vwcvtu_x_x_v_u16m1(nl, vl);
          vuint16m1_t nr16 = __riscv_vwcvtu_x_x_v_u16m1(nr, vl);

          // Reinterpret as signed for arithmetic
          vint16m1_t pl_s = __riscv_vreinterpret_v_u16m1_i16m1(pl16);
          vint16m1_t pr_s = __riscv_vreinterpret_v_u16m1_i16m1(pr16);
          vint16m1_t cl_s = __riscv_vreinterpret_v_u16m1_i16m1(cl16);
          vint16m1_t cr_s = __riscv_vreinterpret_v_u16m1_i16m1(cr16);
          vint16m1_t nl_s = __riscv_vreinterpret_v_u16m1_i16m1(nl16);
          vint16m1_t nr_s = __riscv_vreinterpret_v_u16m1_i16m1(nr16);

          // Gx = (pr - pl) + 2*(cr - cl) + (nr - nl)
          vint16m1_t d_prev = __riscv_vsub_vv_i16m1(pr_s, pl_s, vl);
          vint16m1_t d_curr = __riscv_vsub_vv_i16m1(cr_s, cl_s, vl);
          vint16m1_t d_next = __riscv_vsub_vv_i16m1(nr_s, nl_s, vl);

          // result = d_prev + 2*d_curr + d_next
          vint16m1_t two_d_curr = __riscv_vsll_vx_i16m1(d_curr, 1, vl);
          vint16m1_t sum = __riscv_vadd_vv_i16m1(d_prev, two_d_curr, vl);
          sum = __riscv_vadd_vv_i16m1(sum, d_next, vl);

          __riscv_vse16_v_i16m1(dst_row + x, sum, vl);
          x += vl;
        }
      }

      // Right border: scalar for x in [total_width - channels, total_width)
      for (size_t x = total_width - channels; x < total_width; ++x) {
        dst_row[x] = sobel_horiz_scalar(row_prev, row_curr, row_next, x,
                                        channels, total_width);
      }
    } else {
      // Entire row is border: use scalar
      for (size_t x = 0; x < total_width; ++x) {
        dst_row[x] = sobel_horiz_scalar(row_prev, row_curr, row_next, x,
                                        channels, total_width);
      }
    }
#else
    // Full scalar path
    for (size_t x = 0; x < total_width; ++x) {
      dst_row[x] = sobel_horiz_scalar(row_prev, row_curr, row_next, x,
                                      channels, total_width);
    }
#endif
  }
  return KLEIDICV_OK;
}

// Sobel 3x3 vertical: computes dI/dy using the Sobel operator.
// Kernel: [-1 -2 -1; 0 0 0; 1 2 1]
KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t sobel_3x3_vertical_stripe_s16_u8(
    const uint8_t *src, size_t src_stride, int16_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end,
    size_t channels) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  size_t total_width = width * channels;

  for (size_t y = y_begin; y < y_end; ++y) {
    const uint8_t *row_prev =
        row_ptr(src, src_stride, (y == 0) ? 0 : y - 1);
    // row_curr is not needed for vertical Sobel (middle row coefficients are 0)
    const uint8_t *row_next =
        row_ptr(src, src_stride, (y + 1 >= height) ? height - 1 : y + 1);
    int16_t *dst_row = row_ptr(dst, dst_stride, y);

#ifdef __riscv_vector
    if (total_width > 2 * channels) {
      // Left border: scalar for x in [0, channels)
      for (size_t x = 0; x < channels; ++x) {
        dst_row[x] =
            sobel_vert_scalar(row_prev, row_next, x, channels, total_width);
      }

      // Interior: RVV for x in [channels, total_width - channels)
      // Bounds analysis identical to horizontal case above.
      {
        size_t x = channels;
        size_t end = total_width - channels;
        while (x < end) {
          size_t vl = __riscv_vsetvl_e16m1(end - x);

          // Load left (x - channels), center (x), right (x + channels)
          // from prev and next rows
          vuint8mf2_t pl = __riscv_vle8_v_u8mf2(row_prev + x - channels, vl);
          vuint8mf2_t pc = __riscv_vle8_v_u8mf2(row_prev + x, vl);
          vuint8mf2_t pr = __riscv_vle8_v_u8mf2(row_prev + x + channels, vl);
          vuint8mf2_t nl = __riscv_vle8_v_u8mf2(row_next + x - channels, vl);
          vuint8mf2_t nc = __riscv_vle8_v_u8mf2(row_next + x, vl);
          vuint8mf2_t nr = __riscv_vle8_v_u8mf2(row_next + x + channels, vl);

          // Widen u8 -> u16
          vuint16m1_t pl16 = __riscv_vwcvtu_x_x_v_u16m1(pl, vl);
          vuint16m1_t pc16 = __riscv_vwcvtu_x_x_v_u16m1(pc, vl);
          vuint16m1_t pr16 = __riscv_vwcvtu_x_x_v_u16m1(pr, vl);
          vuint16m1_t nl16 = __riscv_vwcvtu_x_x_v_u16m1(nl, vl);
          vuint16m1_t nc16 = __riscv_vwcvtu_x_x_v_u16m1(nc, vl);
          vuint16m1_t nr16 = __riscv_vwcvtu_x_x_v_u16m1(nr, vl);

          // Reinterpret as signed
          vint16m1_t pl_s = __riscv_vreinterpret_v_u16m1_i16m1(pl16);
          vint16m1_t pc_s = __riscv_vreinterpret_v_u16m1_i16m1(pc16);
          vint16m1_t pr_s = __riscv_vreinterpret_v_u16m1_i16m1(pr16);
          vint16m1_t nl_s = __riscv_vreinterpret_v_u16m1_i16m1(nl16);
          vint16m1_t nc_s = __riscv_vreinterpret_v_u16m1_i16m1(nc16);
          vint16m1_t nr_s = __riscv_vreinterpret_v_u16m1_i16m1(nr16);

          // Gy = (nl + 2*nc + nr) - (pl + 2*pc + pr)
          // Compute prev_sum = pl + 2*pc + pr
          vint16m1_t two_pc = __riscv_vsll_vx_i16m1(pc_s, 1, vl);
          vint16m1_t prev_sum = __riscv_vadd_vv_i16m1(pl_s, two_pc, vl);
          prev_sum = __riscv_vadd_vv_i16m1(prev_sum, pr_s, vl);

          // Compute next_sum = nl + 2*nc + nr
          vint16m1_t two_nc = __riscv_vsll_vx_i16m1(nc_s, 1, vl);
          vint16m1_t next_sum = __riscv_vadd_vv_i16m1(nl_s, two_nc, vl);
          next_sum = __riscv_vadd_vv_i16m1(next_sum, nr_s, vl);

          // result = next_sum - prev_sum
          vint16m1_t result = __riscv_vsub_vv_i16m1(next_sum, prev_sum, vl);

          __riscv_vse16_v_i16m1(dst_row + x, result, vl);
          x += vl;
        }
      }

      // Right border: scalar for x in [total_width - channels, total_width)
      for (size_t x = total_width - channels; x < total_width; ++x) {
        dst_row[x] =
            sobel_vert_scalar(row_prev, row_next, x, channels, total_width);
      }
    } else {
      // Entire row is border: use scalar
      for (size_t x = 0; x < total_width; ++x) {
        dst_row[x] =
            sobel_vert_scalar(row_prev, row_next, x, channels, total_width);
      }
    }
#else
    // Full scalar path
    for (size_t x = 0; x < total_width; ++x) {
      dst_row[x] =
          sobel_vert_scalar(row_prev, row_next, x, channels, total_width);
    }
#endif
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::neon
