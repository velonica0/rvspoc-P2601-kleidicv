// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdlib>
#include <cstring>

#include "kleidicv/rvv.h"
#include "kleidicv/workspace/border_types.h"
#include "median_blur_border_handling.h"

namespace kleidicv::neon {

// Histogram-based median blur for uint8_t (small kernels) with optional
// RVV vectorization.
//
// Uses a sliding-window two-level histogram (coarse[16] + fine[16][16])
// with uint8_t counters.  Per-column histograms are maintained and
// bulk-added / bulk-subtracted as the window slides horizontally,
// exposing 16-element vector add/sub operations that RVV can accelerate.

namespace {

// ---------------------------------------------------------------------------
// Bulk histogram helpers — vectorized under __riscv_vector, scalar otherwise.
// Each operates on a contiguous array of 16 uint8_t values.
// ---------------------------------------------------------------------------

inline void hist16_add_u8(uint8_t* dst, const uint8_t* src) {
#ifdef __riscv_vector
  size_t b = 0;
  while (b < 16) {
    size_t vl = __riscv_vsetvl_e8m1(16 - b);
    vuint8m1_t vd = __riscv_vle8_v_u8m1(dst + b, vl);
    vuint8m1_t vs = __riscv_vle8_v_u8m1(src + b, vl);
    __riscv_vse8_v_u8m1(dst + b, __riscv_vadd_vv_u8m1(vd, vs, vl), vl);
    b += vl;
  }
#else
  for (size_t i = 0; i < 16; ++i) dst[i] += src[i];
#endif
}

inline void hist16_sub_u8(uint8_t* dst, const uint8_t* src) {
#ifdef __riscv_vector
  size_t b = 0;
  while (b < 16) {
    size_t vl = __riscv_vsetvl_e8m1(16 - b);
    vuint8m1_t vd = __riscv_vle8_v_u8m1(dst + b, vl);
    vuint8m1_t vs = __riscv_vle8_v_u8m1(src + b, vl);
    __riscv_vse8_v_u8m1(dst + b, __riscv_vsub_vv_u8m1(vd, vs, vl), vl);
    b += vl;
  }
#else
  for (size_t i = 0; i < 16; ++i) dst[i] -= src[i];
#endif
}

// Add/subtract for a full 256-element fine histogram (uint8_t counters).
inline void hist256_add_u8(uint8_t* dst, const uint8_t* src) {
#ifdef __riscv_vector
  size_t b = 0;
  while (b < 256) {
    size_t vl = __riscv_vsetvl_e8m1(256 - b);
    vuint8m1_t vd = __riscv_vle8_v_u8m1(dst + b, vl);
    vuint8m1_t vs = __riscv_vle8_v_u8m1(src + b, vl);
    __riscv_vse8_v_u8m1(dst + b, __riscv_vadd_vv_u8m1(vd, vs, vl), vl);
    b += vl;
  }
#else
  for (size_t i = 0; i < 256; ++i) dst[i] += src[i];
#endif
}

inline void hist256_sub_u8(uint8_t* dst, const uint8_t* src) {
#ifdef __riscv_vector
  size_t b = 0;
  while (b < 256) {
    size_t vl = __riscv_vsetvl_e8m1(256 - b);
    vuint8m1_t vd = __riscv_vle8_v_u8m1(dst + b, vl);
    vuint8m1_t vs = __riscv_vle8_v_u8m1(src + b, vl);
    __riscv_vse8_v_u8m1(dst + b, __riscv_vsub_vv_u8m1(vd, vs, vl), vl);
    b += vl;
  }
#else
  for (size_t i = 0; i < 256; ++i) dst[i] -= src[i];
#endif
}

// Zero a contiguous block of uint8_t values using RVV where available.
inline void hist_zero_u8(uint8_t* dst, size_t count) {
#ifdef __riscv_vector
  size_t b = 0;
  while (b < count) {
    size_t vl = __riscv_vsetvl_e8m1(count - b);
    __riscv_vse8_v_u8m1(dst + b, __riscv_vmv_v_x_u8m1(0, vl), vl);
    b += vl;
  }
#else
  std::memset(dst, 0, count);
#endif
}

}  // namespace

// ---------------------------------------------------------------------------
// The algorithm maintains per-column two-level histograms:
//   col_coarse[col][16]      — counts for each coarse bin (pixel >> 4)
//   col_fine[col][256]       — counts for each fine value
//
// For each output pixel the window histogram is built by summing ksize
// column histograms.  As the window slides right by one column, one
// column histogram is added and one is subtracted — a bulk 16-element
// (coarse) or 256-element (fine) vector operation that RVV accelerates.
// ---------------------------------------------------------------------------

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t median_blur_small_hist_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, size_t kernel_height, FixedBorderType border_type) {
  const size_t ksize = kernel_height;
  const size_t kMargin = ksize / 2;
  const size_t target_cdf = (ksize * ksize) / 2;

  Rows<const uint8_t> src_rows{src, src_stride, channels};
  Rows<uint8_t> dst_rows{dst, dst_stride, channels};

  const size_t total_cols = width + 2 * kMargin;  // cols including border

  // Allocate per-column histograms.
  // col_coarse: total_cols * 16 uint8_t
  // col_fine:   total_cols * 256 uint8_t
  const size_t coarse_buf_size = total_cols * 16;
  const size_t fine_buf_size = total_cols * 256;
  uint8_t *col_buf = static_cast<uint8_t *>(
      std::malloc(coarse_buf_size + fine_buf_size));
  if (!col_buf) return KLEIDICV_ERROR_ALLOCATION;

  uint8_t *col_coarse = col_buf;
  uint8_t *col_fine   = col_buf + coarse_buf_size;

  for (size_t ch = 0; ch < channels; ++ch) {
    // Zero column histograms.
    hist_zero_u8(col_coarse, coarse_buf_size);
    hist_zero_u8(col_fine, fine_buf_size);

    // Initialize column histograms for the first output row's window.
    for (size_t r = 0; r < ksize; ++r) {
      const ptrdiff_t valid_h =
          get_physical_index(y_begin + r - kMargin, height, border_type);
      for (size_t tc = 0; tc < total_cols; ++tc) {
        const ptrdiff_t valid_w =
            get_physical_index(tc - kMargin, width, border_type);
        uint8_t pixel =
            src_rows.at(valid_h, valid_w)[static_cast<ptrdiff_t>(ch)];
        col_coarse[tc * 16 + (pixel >> 4)]++;
        col_fine[tc * 256 + pixel]++;
      }
    }

    // Process each output row.
    for (size_t h = y_begin; h < y_end; ++h) {
      // If not the first row, update column histograms: add new bottom
      // row, remove old top row.
      if (h != y_begin) {
        const ptrdiff_t valid_new_h =
            get_physical_index(h + kMargin, height, border_type);
        const ptrdiff_t valid_old_h =
            get_physical_index(h - kMargin - 1, height, border_type);
        for (size_t tc = 0; tc < total_cols; ++tc) {
          const ptrdiff_t valid_w =
              get_physical_index(tc - kMargin, width, border_type);
          uint8_t incoming =
              src_rows.at(valid_new_h, valid_w)[static_cast<ptrdiff_t>(ch)];
          uint8_t outgoing =
              src_rows.at(valid_old_h, valid_w)[static_cast<ptrdiff_t>(ch)];
          if (incoming != outgoing) {
            col_coarse[tc * 16 + (incoming >> 4)]++;
            col_coarse[tc * 16 + (outgoing >> 4)]--;
            col_fine[tc * 256 + incoming]++;
            col_fine[tc * 256 + outgoing]--;
          }
        }
      }

      // Build the window histogram for the first output column (w = 0)
      // by summing ksize column histograms.
      uint8_t win_coarse[16];
      uint8_t win_fine[256];
      std::memset(win_coarse, 0, sizeof(win_coarse));
      std::memset(win_fine, 0, sizeof(win_fine));

      for (size_t tc = 0; tc < ksize; ++tc) {
        hist16_add_u8(win_coarse, &col_coarse[tc * 16]);
        hist256_add_u8(win_fine, &col_fine[tc * 256]);
      }

      // Find median for w = 0.
      {
        uint8_t cum = 0;
        int ci = 0;
        while (true) {
          if ((cum + win_coarse[ci]) > target_cdf) break;
          cum += win_coarse[ci];
          ci++;
        }
        int fi = ci * 16;
        while (true) {
          cum += win_fine[fi];
          if (cum > target_cdf) break;
          fi++;
        }
        dst_rows.at(static_cast<ptrdiff_t>(h), 0)[static_cast<ptrdiff_t>(ch)] =
            static_cast<uint8_t>(fi);
      }

      // Slide horizontally for remaining columns.
      for (size_t w = 1; w < width; ++w) {
        // Remove the leftmost column of the old window.
        size_t remove_tc = w - 1;
        hist16_sub_u8(win_coarse, &col_coarse[remove_tc * 16]);
        hist256_sub_u8(win_fine, &col_fine[remove_tc * 256]);

        // Add the new rightmost column.
        size_t add_tc = w + ksize - 1;
        hist16_add_u8(win_coarse, &col_coarse[add_tc * 16]);
        hist256_add_u8(win_fine, &col_fine[add_tc * 256]);

        // Find median.
        uint8_t cum = 0;
        int ci = 0;
        while (true) {
          if ((cum + win_coarse[ci]) > target_cdf) break;
          cum += win_coarse[ci];
          ci++;
        }
        int fi = ci * 16;
        while (true) {
          cum += win_fine[fi];
          if (cum > target_cdf) break;
          fi++;
        }
        dst_rows.at(static_cast<ptrdiff_t>(h),
                    static_cast<ptrdiff_t>(w))[static_cast<ptrdiff_t>(ch)] =
            static_cast<uint8_t>(fi);
      }
    }
  }

  std::free(col_buf);
  return KLEIDICV_OK;
}

}  // namespace kleidicv::neon
