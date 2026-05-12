// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

// Scalar implementation of fixed-kernel Gaussian blur for the RISC-V port.
// This replicates the same fixed-point arithmetic that the Neon version uses
// so that output values match numerically.

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

// --- Border index helpers ---

// Map a coordinate into [0, size-1] according to the given border type.
static inline ptrdiff_t border_idx(ptrdiff_t idx, ptrdiff_t size,
                                   FixedBorderType bt) {
  if (idx >= 0 && idx < size) return idx;

  switch (bt) {
    case FixedBorderType::REPLICATE:
      return (idx < 0) ? 0 : size - 1;

    case FixedBorderType::REFLECT:
      // Neon convention: REFLECT = OpenCV BORDER_REFLECT (edge pixel repeated)
      // Pattern: dcba|abcde|edcba  => idx -1 -> 0, -2 -> 1, ...
      if (size == 1) return 0;
      {
        ptrdiff_t period = 2 * size;
        ptrdiff_t p = idx < 0 ? -(idx + 1) : idx;
        p %= period;
        return (p < size) ? p : period - 1 - p;
      }

    case FixedBorderType::WRAP:
      {
        ptrdiff_t p = idx % size;
        return (p < 0) ? p + size : p;
      }

    case FixedBorderType::REVERSE:
      // Neon convention: REVERSE = OpenCV BORDER_REFLECT_101 (edge pixel NOT repeated)
      // Pattern: dcb|abcde|dcb  => idx -1 -> 1, -2 -> 2, ...
      if (size == 1) return 0;
      {
        ptrdiff_t period = 2 * (size - 1);
        ptrdiff_t p = idx < 0 ? -idx : idx;
        p %= period;
        return (p < size) ? p : period - p;
      }

    default:
      return (idx < 0) ? 0 : size - 1;
  }
}

// Rounding shift right matching the neon version's rounding_shift_right.
static inline uint8_t rshift_round(uint32_t val, unsigned shift) {
  return static_cast<uint8_t>((val + (1u << (shift - 1))) >> shift);
}

// ===== Binomial kernel implementation (sigma == 0) =====

// Binomial 1D kernel weights.  The vertical pass accumulates into uint16_t
// using these weights; the horizontal pass uses the same weights on uint16_t
// intermediate values accumulated into uint32_t, then rounds-right by
// total_shift.

struct BinomialKernel {
  const uint16_t *weights;
  size_t size;
  unsigned total_shift;  // combined shift for the full 2D normalization
};

// 3x3: [1, 2, 1], normalizer = 4 * 4 = 16, shift = 4
static const uint16_t binom3[] = {1, 2, 1};
// 5x5: [1, 4, 6, 4, 1], normalizer = 16 * 16 = 256, shift = 8
static const uint16_t binom5[] = {1, 4, 6, 4, 1};
// 7x7: [2, 7, 14, 18, 14, 7, 2], normalizer = 64 * 64 = 4096, shift = 12
static const uint16_t binom7[] = {2, 7, 14, 18, 14, 7, 2};
// 9x9: [4, 13, 30, 51, 60, 51, 30, 13, 4], normalizer = 256*256 = 65536,
// shift = 16
static const uint16_t binom9[] = {4, 13, 30, 51, 60, 51, 30, 13, 4};

static const BinomialKernel binomial_kernels[] = {
    {binom3, 3, 4},
    {binom5, 5, 8},
    {binom7, 7, 12},
    {binom9, 9, 16},
};

static const BinomialKernel *get_binomial_kernel(size_t kernel_size) {
  switch (kernel_size) {
    case 3: return &binomial_kernels[0];
    case 5: return &binomial_kernels[1];
    case 7: return &binomial_kernels[2];
    case 9: return &binomial_kernels[3];
    default: return nullptr;
  }
}

// Perform a binomial Gaussian blur using separable integer convolution.
// The vertical pass convolves each column into a uint16_t temp buffer, then
// the horizontal pass convolves horizontally and rounds to uint8_t.
// The caller provides the pre-allocated tmp buffer (uint16_t, row_elems entries).
static void gaussian_blur_binomial(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels,
    const BinomialKernel &bk, FixedBorderType border_type, uint16_t *tmp) {
  const ptrdiff_t half = static_cast<ptrdiff_t>(bk.size / 2);
  const ptrdiff_t w = static_cast<ptrdiff_t>(width);
  const ptrdiff_t h = static_cast<ptrdiff_t>(height);
  const size_t row_elems = width * channels;

  for (size_t y = y_begin; y < y_end; ++y) {
    // --- Vertical pass ---
#ifdef __riscv_vector
    {
      size_t x = 0;
      while (x < row_elems) {
        size_t vl = __riscv_vsetvl_e16m1(row_elems - x);
        vuint16m1_t acc = __riscv_vmv_v_x_u16m1(0, vl);
        for (ptrdiff_t ky = -half; ky <= half; ++ky) {
          ptrdiff_t sy = border_idx(static_cast<ptrdiff_t>(y) + ky, h,
                                    border_type);
          const uint8_t *src_row =
              row_ptr(src, src_stride, static_cast<size_t>(sy));
          // Load u8 elements, widen to u16, multiply-accumulate
          vuint8mf2_t v8 = __riscv_vle8_v_u8mf2(src_row + x, vl);
          vuint16m1_t v16 = __riscv_vzext_vf2_u16m1(v8, vl);
          acc = __riscv_vmacc_vx_u16m1(acc, bk.weights[ky + half], v16, vl);
        }
        __riscv_vse16_v_u16m1(tmp + x, acc, vl);
        x += vl;
      }
    }
#else
    for (size_t x_elem = 0; x_elem < row_elems; ++x_elem) {
      uint32_t acc = 0;
      for (ptrdiff_t ky = -half; ky <= half; ++ky) {
        ptrdiff_t sy = border_idx(static_cast<ptrdiff_t>(y) + ky, h,
                                  border_type);
        const uint8_t *src_row =
            row_ptr(src, src_stride, static_cast<size_t>(sy));
        acc += static_cast<uint32_t>(src_row[x_elem]) *
               bk.weights[ky + half];
      }
      tmp[x_elem] = static_cast<uint16_t>(acc);
    }
#endif

    // --- Horizontal pass ---
    uint8_t *dst_row = row_ptr(dst, dst_stride, y);
    for (size_t px = 0; px < width; ++px) {
      for (size_t ch = 0; ch < channels; ++ch) {
        uint32_t acc = 0;
        for (ptrdiff_t kx = -half; kx <= half; ++kx) {
          ptrdiff_t sx = border_idx(static_cast<ptrdiff_t>(px) + kx, w,
                                    border_type);
          acc += static_cast<uint32_t>(
                     tmp[static_cast<size_t>(sx) * channels + ch]) *
                 bk.weights[kx + half];
        }
        dst_row[px * channels + ch] = rshift_round(acc, bk.total_shift);
      }
    }
  }
}

// ===== Non-binomial (sigma != 0) or larger fixed kernels =====
// Uses generate_gaussian_half_kernel to produce uint8_t half-kernel
// coefficients, then applies two-pass fixed-point separable convolution
// matching the neon scalar path.

// The caller provides the pre-allocated tmp buffer (uint8_t, row_elems entries).
static void gaussian_blur_half_kernel(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_size, float sigma, FixedBorderType border_type,
    uint8_t *tmp) {
  const size_t half_kernel_size = get_half_kernel_size(kernel_size);
  const ptrdiff_t w = static_cast<ptrdiff_t>(width);
  const ptrdiff_t h = static_cast<ptrdiff_t>(height);
  const size_t row_elems = width * channels;

  uint8_t half_kern[128];
  bool success = generate_gaussian_half_kernel(half_kern, half_kernel_size,
                                               sigma);
  if (!success) {
    // Sigma is so small the kernel is a delta => just copy.
    for (size_t y = y_begin; y < y_end; ++y) {
      const uint8_t *src_row = row_ptr(src, src_stride, y);
      uint8_t *dst_row = row_ptr(dst, dst_stride, y);
      std::memcpy(dst_row, src_row, row_elems);
    }
    return;
  }

  // half_kern layout: index 0 is the outermost weight, index
  // half_kernel_size-1 is the center weight.  The convolution pairs
  // symmetric positions and multiplies by half_kern[i].

  for (size_t y = y_begin; y < y_end; ++y) {
    // --- Vertical pass (produces uint8_t via rounding shift right by 8) ---
#ifdef __riscv_vector
    {
      // Pre-compute row pointers for center and symmetric pairs.
      ptrdiff_t center_y = border_idx(static_cast<ptrdiff_t>(y), h,
                                       border_type);
      const uint8_t *center_row =
          row_ptr(src, src_stride, static_cast<size_t>(center_y));

      // Pre-compute symmetric row pointers to avoid recomputing per-vector.
      const uint8_t *rows_neg[128];
      const uint8_t *rows_pos[128];
      for (size_t i = 0; i < half_kernel_size - 1; ++i) {
        ptrdiff_t ky_neg = static_cast<ptrdiff_t>(y) -
                           static_cast<ptrdiff_t>(half_kernel_size - 1 - i);
        ptrdiff_t ky_pos = static_cast<ptrdiff_t>(y) +
                           static_cast<ptrdiff_t>(half_kernel_size - 1 - i);
        ptrdiff_t sy_neg = border_idx(ky_neg, h, border_type);
        ptrdiff_t sy_pos = border_idx(ky_pos, h, border_type);
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
        __riscv_vse8_v_u8m1(tmp + x, result, vl);
        x += vl;
      }
    }
#else
    for (size_t x_elem = 0; x_elem < row_elems; ++x_elem) {
      // Replicate the neon scalar path exactly:
      // acc = src[center] * half_kern[half_kernel_size-1]
      // + sum_i (src[i] + src[mirror_i]) * half_kern[i]
      ptrdiff_t center_y = border_idx(static_cast<ptrdiff_t>(y), h,
                                       border_type);
      const uint8_t *center_row =
          row_ptr(src, src_stride, static_cast<size_t>(center_y));
      uint32_t acc = static_cast<uint32_t>(center_row[x_elem]) *
                     half_kern[half_kernel_size - 1];

      for (size_t i = 0; i < half_kernel_size - 1; ++i) {
        ptrdiff_t ky_neg = static_cast<ptrdiff_t>(y) -
                           static_cast<ptrdiff_t>(half_kernel_size - 1 - i);
        ptrdiff_t ky_pos = static_cast<ptrdiff_t>(y) +
                           static_cast<ptrdiff_t>(half_kernel_size - 1 - i);
        ptrdiff_t sy_neg = border_idx(ky_neg, h, border_type);
        ptrdiff_t sy_pos = border_idx(ky_pos, h, border_type);
        const uint8_t *row_neg =
            row_ptr(src, src_stride, static_cast<size_t>(sy_neg));
        const uint8_t *row_pos =
            row_ptr(src, src_stride, static_cast<size_t>(sy_pos));
        acc += (static_cast<uint32_t>(row_neg[x_elem]) +
                static_cast<uint32_t>(row_pos[x_elem])) *
               half_kern[i];
      }

      tmp[x_elem] = rshift_round(acc, 8);
    }
#endif

    // --- Horizontal pass (same fixed-point approach) ---
    uint8_t *dst_row = row_ptr(dst, dst_stride, y);
    for (size_t px = 0; px < width; ++px) {
      for (size_t ch = 0; ch < channels; ++ch) {
        size_t center_idx = px * channels + ch;
        uint32_t acc = static_cast<uint32_t>(tmp[center_idx]) *
                       half_kern[half_kernel_size - 1];

        for (size_t i = 0; i < half_kernel_size - 1; ++i) {
          ptrdiff_t kx_neg = static_cast<ptrdiff_t>(px) -
                             static_cast<ptrdiff_t>(half_kernel_size - 1 - i);
          ptrdiff_t kx_pos = static_cast<ptrdiff_t>(px) +
                             static_cast<ptrdiff_t>(half_kernel_size - 1 - i);
          ptrdiff_t sx_neg = border_idx(kx_neg, w, border_type);
          ptrdiff_t sx_pos = border_idx(kx_pos, w, border_type);
          acc += (static_cast<uint32_t>(
                      tmp[static_cast<size_t>(sx_neg) * channels + ch]) +
                  static_cast<uint32_t>(
                      tmp[static_cast<size_t>(sx_pos) * channels + ch])) *
                 half_kern[i];
        }

        dst_row[px * channels + ch] = rshift_round(acc, 8);
      }
    }
  }
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t gaussian_blur_fixed_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, size_t /*kernel_height*/, float sigma_x,
    float /*sigma_y*/, FixedBorderType fixed_border_type) {
  if (auto result =
          gaussian_blur_checks(src, src_stride, dst, dst_stride, width, height);
      result != KLEIDICV_OK) {
    return result;
  }

  const size_t row_elems = width * channels;

  if (sigma_x == 0.0f) {
    // Binomial kernel (sigma==0 means use binomial approximation).
    const BinomialKernel *bk = get_binomial_kernel(kernel_width);
    if (bk) {
      // Allocate the tmp buffer for the binomial vertical pass (uint16_t).
      uint16_t *tmp = static_cast<uint16_t *>(
          std::malloc(row_elems * sizeof(uint16_t)));
      if (!tmp) {
        return KLEIDICV_ERROR_ALLOCATION;
      }
      gaussian_blur_binomial(src, src_stride, dst, dst_stride, width, height,
                             y_begin, y_end, channels, *bk,
                             fixed_border_type, tmp);
      std::free(tmp);
      return KLEIDICV_OK;
    }
    // Fall through to half-kernel path for sizes > 9 (15, 21).
  }

  // Allocate the tmp buffer for the half-kernel vertical pass (uint8_t).
  uint8_t *tmp = static_cast<uint8_t *>(std::malloc(row_elems));
  if (!tmp) {
    return KLEIDICV_ERROR_ALLOCATION;
  }
  gaussian_blur_half_kernel(src, src_stride, dst, dst_stride, width, height,
                            y_begin, y_end, channels, kernel_width, sigma_x,
                            fixed_border_type, tmp);
  std::free(tmp);
  return KLEIDICV_OK;
}

}  // namespace kleidicv::neon
