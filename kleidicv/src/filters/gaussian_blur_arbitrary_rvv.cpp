// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Scalar implementation of arbitrary-kernel Gaussian blur for the RISC-V port.
// This replicates the same fixed-point arithmetic that the Neon version uses
// (generate_gaussian_half_kernel + rounding_shift_right by 8 per pass) so that
// output values match numerically.

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <cstring>
#include <vector>

#include "kleidicv/filters/gaussian_blur.h"
#include "kleidicv/filters/sigma.h"
#include "kleidicv/rvv.h"
#include "kleidicv/workspace/border_types.h"

namespace kleidicv::neon {

// --- Border index helper ---

static inline ptrdiff_t arb_border_idx(ptrdiff_t idx, ptrdiff_t size,
                                       FixedBorderType bt) {
  if (idx >= 0 && idx < size) return idx;

  switch (bt) {
    case FixedBorderType::REPLICATE:
      return (idx < 0) ? 0 : size - 1;

    case FixedBorderType::REFLECT:
      if (size == 1) return 0;
      {
        ptrdiff_t period = 2 * (size - 1);
        ptrdiff_t p = idx < 0 ? -idx : idx;
        p %= period;
        return (p < size) ? p : period - p;
      }

    case FixedBorderType::WRAP:
      {
        ptrdiff_t p = idx % size;
        return (p < 0) ? p + size : p;
      }

    case FixedBorderType::REVERSE:
      if (size == 1) return 0;
      {
        ptrdiff_t period = 2 * size;
        ptrdiff_t p = idx < 0 ? -(idx + 1) : idx;
        p %= period;
        return (p < size) ? p : period - 1 - p;
      }

    default:
      return (idx < 0) ? 0 : size - 1;
  }
}

// Rounding shift right matching rounding_shift_right in utils.h.
static inline uint8_t arb_rshift_round(uint32_t val, unsigned shift) {
  return static_cast<uint8_t>((val + (1u << (shift - 1))) >> shift);
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t gaussian_blur_arbitrary_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, size_t /*kernel_height*/, float sigma_x,
    float /*sigma_y*/, FixedBorderType fixed_border_type) {
  if (auto result =
          gaussian_blur_checks(src, src_stride, dst, dst_stride, width, height);
      result != KLEIDICV_OK) {
    return result;
  }

  const size_t half_kernel_size = get_half_kernel_size(kernel_width);
  const ptrdiff_t w = static_cast<ptrdiff_t>(width);
  const ptrdiff_t h = static_cast<ptrdiff_t>(height);
  const size_t row_elems = width * channels;

  // Generate the half-kernel using the same function the neon code uses.
  uint8_t half_kern[128];
  bool success = generate_gaussian_half_kernel(half_kern, half_kernel_size,
                                               sigma_x);
  if (!success) {
    // Sigma is so small the kernel is a delta => just copy.
    for (size_t y = y_begin; y < y_end; ++y) {
      const uint8_t *src_row = row_ptr(src, src_stride, y);
      uint8_t *dst_row = row_ptr(dst, dst_stride, y);
      std::memcpy(dst_row, src_row, row_elems);
    }
    return KLEIDICV_OK;
  }

  // Temporary row buffer for vertical pass output (uint8_t).
  // The neon arbitrary path uses uint8_t intermediate buffer, with each pass
  // independently applying rounding_shift_right by 8.
  std::vector<uint8_t> tmp(row_elems);

  for (size_t y = y_begin; y < y_end; ++y) {
    // --- Vertical pass ---
    // Replicate the neon arbitrary scalar vertical path:
    //   acc = src[center] * half_kern[half_kernel_size-1]
    //   for i in 0..half_kernel_size-2:
    //     acc += (src[y - (hks-1-i)] + src[y + (hks-1-i)]) * half_kern[i]
    //   result = rounding_shift_right(acc, 8)
#ifdef __riscv_vector
    {
      // Pre-compute row pointers for center and symmetric pairs.
      ptrdiff_t center_y = static_cast<ptrdiff_t>(y);
      ptrdiff_t sy_center = arb_border_idx(center_y, h, fixed_border_type);
      const uint8_t *center_row =
          row_ptr(src, src_stride, static_cast<size_t>(sy_center));

      // Pre-compute symmetric row pointers to avoid recomputing per-vector.
      const uint8_t *rows_neg[128];
      const uint8_t *rows_pos[128];
      for (size_t i = 0; i < half_kernel_size - 1; ++i) {
        ptrdiff_t offset =
            static_cast<ptrdiff_t>(half_kernel_size - 1 - i);
        ptrdiff_t sy_neg = arb_border_idx(center_y - offset, h,
                                          fixed_border_type);
        ptrdiff_t sy_pos = arb_border_idx(center_y + offset, h,
                                          fixed_border_type);
        rows_neg[i] = row_ptr(src, src_stride, static_cast<size_t>(sy_neg));
        rows_pos[i] = row_ptr(src, src_stride, static_cast<size_t>(sy_pos));
      }

      size_t x = 0;
      while (x < row_elems) {
        size_t vl = __riscv_vsetvl_e8m1(row_elems - x);

        // Load center row, widening multiply to u16 for accumulation.
        vuint8m1_t vc = __riscv_vle8_v_u8m1(center_row + x, vl);
        vuint16m2_t acc = __riscv_vwmulu_vx_u16m2(
            vc, half_kern[half_kernel_size - 1], vl);

        for (size_t i = 0; i < half_kernel_size - 1; ++i) {
          // Load symmetric pair rows.
          vuint8m1_t v_neg = __riscv_vle8_v_u8m1(rows_neg[i] + x, vl);
          vuint8m1_t v_pos = __riscv_vle8_v_u8m1(rows_pos[i] + x, vl);
          // Widening add: u8 + u8 -> u16
          vuint16m2_t pair_sum = __riscv_vwaddu_vv_u16m2(v_neg, v_pos, vl);
          // Multiply pair_sum by weight and accumulate.
          vuint16m2_t weighted = __riscv_vmul_vx_u16m2(
              pair_sum, static_cast<uint16_t>(half_kern[i]), vl);
          acc = __riscv_vadd_vv_u16m2(acc, weighted, vl);
        }

        // Rounding shift right by 8: (acc + 128) >> 8, then narrow to u8.
        vuint16m2_t rounded = __riscv_vadd_vx_u16m2(acc, 128, vl);
        vuint8m1_t result = __riscv_vnsrl_wx_u8m1(rounded, 8, vl);
        __riscv_vse8_v_u8m1(tmp.data() + x, result, vl);
        x += vl;
      }
    }
#else
    for (size_t x_elem = 0; x_elem < row_elems; ++x_elem) {
      ptrdiff_t center_y = static_cast<ptrdiff_t>(y);
      ptrdiff_t sy_center = arb_border_idx(center_y, h, fixed_border_type);
      const uint8_t *center_row =
          row_ptr(src, src_stride, static_cast<size_t>(sy_center));
      uint32_t acc = static_cast<uint32_t>(center_row[x_elem]) *
                     half_kern[half_kernel_size - 1];

      for (size_t i = 0; i < half_kernel_size - 1; ++i) {
        ptrdiff_t offset =
            static_cast<ptrdiff_t>(half_kernel_size - 1 - i);
        ptrdiff_t sy_neg = arb_border_idx(center_y - offset, h,
                                          fixed_border_type);
        ptrdiff_t sy_pos = arb_border_idx(center_y + offset, h,
                                          fixed_border_type);
        const uint8_t *row_neg =
            row_ptr(src, src_stride, static_cast<size_t>(sy_neg));
        const uint8_t *row_pos =
            row_ptr(src, src_stride, static_cast<size_t>(sy_pos));
        acc += (static_cast<uint32_t>(row_neg[x_elem]) +
                static_cast<uint32_t>(row_pos[x_elem])) *
               half_kern[i];
      }

      tmp[x_elem] = arb_rshift_round(acc, 8);
    }
#endif

    // --- Horizontal pass ---
    // Same half-kernel approach applied horizontally.
    uint8_t *dst_row = row_ptr(dst, dst_stride, y);
    for (size_t px = 0; px < width; ++px) {
      for (size_t ch = 0; ch < channels; ++ch) {
        size_t center_idx = px * channels + ch;
        uint32_t acc = static_cast<uint32_t>(tmp[center_idx]) *
                       half_kern[half_kernel_size - 1];

        for (size_t i = 0; i < half_kernel_size - 1; ++i) {
          ptrdiff_t offset =
              static_cast<ptrdiff_t>(half_kernel_size - 1 - i);
          ptrdiff_t sx_neg = arb_border_idx(
              static_cast<ptrdiff_t>(px) - offset, w, fixed_border_type);
          ptrdiff_t sx_pos = arb_border_idx(
              static_cast<ptrdiff_t>(px) + offset, w, fixed_border_type);
          acc += (static_cast<uint32_t>(
                      tmp[static_cast<size_t>(sx_neg) * channels + ch]) +
                  static_cast<uint32_t>(
                      tmp[static_cast<size_t>(sx_pos) * channels + ch])) *
                 half_kern[i];
        }

        dst_row[px * channels + ch] = arb_rshift_round(acc, 8);
      }
    }
  }

  return KLEIDICV_OK;
}

}  // namespace kleidicv::neon
