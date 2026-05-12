// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "kleidicv/rvv.h"
#include "kleidicv/workspace/border_types.h"
#include "median_blur_border_handling.h"

namespace kleidicv::neon {

// Scalar blur-and-downsample for image pyramid construction.
//
// Applies a 5x5 Gaussian binomial filter:
//
//              [ 1,  4,  6,  4, 1 ]           [ 1 ]
//              [ 4, 16, 24, 16, 4 ]           [ 4 ]
//  F = 1/256 * [ 6, 24, 36, 24, 6 ] = 1/256 * [ 6 ] * [ 1,  4,  6,  4, 1 ]
//              [ 4, 16, 24, 16, 4 ]           [ 4 ]
//              [ 1,  4,  6,  4, 1 ]           [ 1 ]
//
// Then downsamples by 2 in both dimensions (keeping even rows/columns).
//
// y_begin and y_end are in source image coordinates.
// The output row for source row sy is sy / 2 (only even rows produce output).

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t kleidicv_blur_and_downsample_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    uint8_t *dst, size_t dst_stride, size_t y_begin, size_t y_end,
    size_t channels, FixedBorderType fixed_border_type) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, (src_height + 1) / 2);
  CHECK_IMAGE_SIZE(src_width, src_height);

  const size_t total_src_cols = src_width * channels;

  // Intermediate buffer for vertical pass: one row of uint16_t.
  std::vector<uint16_t> vbuf(total_src_cols);

  // Vertical kernel weights: [1, 4, 6, 4, 1].
  static constexpr int vk[5] = {1, 4, 6, 4, 1};

  // Horizontal kernel weights: [1, 4, 6, 4, 1].
  static constexpr int hk[5] = {1, 4, 6, 4, 1};

  // Align y_begin up to the nearest even row.
  size_t sy_start = (y_begin + 1) & ~static_cast<size_t>(1);  // align_up(y_begin, 2)

  // Process only even source rows, which produce output rows.
  for (size_t sy = sy_start; sy < y_end; sy += 2) {
    const size_t dy = sy / 2;  // output row index

    // --- Vertical pass ---
    // Compute vbuf[x] = sum_k vk[k] * src(sy + k - 2, x)
    // for all x in [0, total_src_cols).
#ifdef __riscv_vector
    // RVV vectorized vertical pass
    {
      // Pre-compute the 5 row pointers (handling border).
      const uint8_t *rows[5];
      for (int k = 0; k < 5; ++k) {
        ptrdiff_t row_idx = get_physical_index(
            static_cast<size_t>(static_cast<ptrdiff_t>(sy) + k - 2),
            src_height, fixed_border_type);
        rows[k] = reinterpret_cast<const uint8_t *>(src) +
                  static_cast<size_t>(row_idx) * src_stride;
      }

      size_t x = 0;
      while (x < total_src_cols) {
        size_t vl = __riscv_vsetvl_e16m1(total_src_cols - x);

        // Load 5 rows as u8, widen to u16, apply weights [1,4,6,4,1].
        vuint8mf2_t r0 = __riscv_vle8_v_u8mf2(rows[0] + x, vl);
        vuint8mf2_t r1 = __riscv_vle8_v_u8mf2(rows[1] + x, vl);
        vuint8mf2_t r2 = __riscv_vle8_v_u8mf2(rows[2] + x, vl);
        vuint8mf2_t r3 = __riscv_vle8_v_u8mf2(rows[3] + x, vl);
        vuint8mf2_t r4 = __riscv_vle8_v_u8mf2(rows[4] + x, vl);

        vuint16m1_t v0 = __riscv_vzext_vf2_u16m1(r0, vl);  // weight 1
        vuint16m1_t v1 = __riscv_vzext_vf2_u16m1(r1, vl);  // weight 4
        vuint16m1_t v2 = __riscv_vzext_vf2_u16m1(r2, vl);  // weight 6
        vuint16m1_t v3 = __riscv_vzext_vf2_u16m1(r3, vl);  // weight 4
        vuint16m1_t v4 = __riscv_vzext_vf2_u16m1(r4, vl);  // weight 1

        // acc = v0*1 + v1*4 + v2*6 + v3*4 + v4*1
        vuint16m1_t acc = v0;
        acc = __riscv_vmacc_vx_u16m1(acc, 4, v1, vl);
        acc = __riscv_vmacc_vx_u16m1(acc, 6, v2, vl);
        acc = __riscv_vmacc_vx_u16m1(acc, 4, v3, vl);
        acc = __riscv_vadd_vv_u16m1(acc, v4, vl);

        __riscv_vse16_v_u16m1(vbuf.data() + x, acc, vl);
        x += vl;
      }
    }
#else
    // Scalar vertical pass
    for (size_t x = 0; x < total_src_cols; ++x) {
      uint16_t acc = 0;
      for (int k = 0; k < 5; ++k) {
        ptrdiff_t row_idx = get_physical_index(
            static_cast<size_t>(static_cast<ptrdiff_t>(sy) + k - 2),
            src_height, fixed_border_type);
        const uint8_t *row_data =
            reinterpret_cast<const uint8_t *>(src) +
            static_cast<size_t>(row_idx) * src_stride;
        acc += static_cast<uint16_t>(vk[k]) *
               static_cast<uint16_t>(row_data[x]);
      }
      vbuf[x] = acc;
    }
#endif

    // --- Horizontal pass with 2x downsampling ---
    // Output column dx corresponds to source column sx = dx * 2.
    // Apply horizontal kernel and divide by 256 (rounding shift right by 8).
    const size_t dst_width = (src_width + 1) / 2;
    for (size_t dx = 0; dx < dst_width; ++dx) {
      const size_t sx = dx * 2;
      for (size_t ch = 0; ch < channels; ++ch) {
        uint32_t acc = 0;
        for (int k = 0; k < 5; ++k) {
          ptrdiff_t col_idx = get_physical_index(
              static_cast<size_t>(static_cast<ptrdiff_t>(sx) + k - 2),
              src_width, fixed_border_type);
          acc += static_cast<uint32_t>(hk[k]) *
                 static_cast<uint32_t>(
                     vbuf[static_cast<size_t>(col_idx) * channels + ch]);
        }
        // Rounding shift right by 8 to divide by 256.
        uint8_t result = static_cast<uint8_t>((acc + 128) >> 8);
        uint8_t *dst_row = reinterpret_cast<uint8_t *>(
            reinterpret_cast<uint8_t *>(dst) + dy * dst_stride);
        dst_row[dx * channels + ch] = result;
      }
    }
  }

  return KLEIDICV_OK;
}

}  // namespace kleidicv::neon
