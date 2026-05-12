// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

// ========== 2x2 u8 upscale ==========
// Matches the Neon implementation exactly:
//   dst[2*sy][2*sx]     = src[sy][sx]  (top/left edges are exact copies)
//   Interpolation uses fixed integer weights:
//     lerp1d(near, far) = (near*3 + far + 2) >> 2
//     lerp2d(near, mid_a, mid_b, far) = (near*9 + (mid_a+mid_b)*3 + far + 8) >> 4

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t kleidicv_resize_2x2_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, uint8_t *dst, size_t dst_stride) {
  void *_ws = std::malloc(src_width * 4);
  if (!_ws) return KLEIDICV_ERROR_ALLOCATION;
  std::free(_ws);

  size_t dst_width = src_width * 2;

  auto lerp1d_scalar = [](uint8_t near, uint8_t far) -> uint8_t {
    return static_cast<uint8_t>((near * 3 + far + 2) >> 2);
  };

  auto lerp2d_scalar = [](uint8_t near, uint8_t mid_a, uint8_t mid_b,
                          uint8_t far) -> uint8_t {
    return static_cast<uint8_t>((near * 9 + (mid_a + mid_b) * 3 + far + 8) >> 4);
  };

  // Handle top or bottom edge row: only horizontal interpolation.
  auto process_edge_row = [src_width, dst_width, lerp1d_scalar](
                              const uint8_t *src_row, uint8_t *dst_row) {
    // Left element: exact copy
    dst_row[0] = src_row[0];

    // Right element: exact copy
    dst_row[dst_width - 1] = src_row[src_width - 1];

    // Middle elements
    for (size_t src_x = 0; src_x + 1 < src_width; ++src_x) {
      size_t dst_x = src_x * 2 + 1;
      const uint8_t src_left = src_row[src_x], src_right = src_row[src_x + 1];
      dst_row[dst_x] = lerp1d_scalar(src_left, src_right);
      dst_row[dst_x + 1] = lerp1d_scalar(src_right, src_left);
    }
  };

  // Handle interior rows: both horizontal and vertical interpolation.
  auto process_row = [src_width, dst_width, lerp1d_scalar, lerp2d_scalar](
                         const uint8_t *src_row0, const uint8_t *src_row1,
                         uint8_t *dst_row0, uint8_t *dst_row1) {
    // Left element: vertical-only interpolation
    dst_row0[0] = lerp1d_scalar(src_row0[0], src_row1[0]);
    dst_row1[0] = lerp1d_scalar(src_row1[0], src_row0[0]);

    // Right element: vertical-only interpolation
    dst_row0[dst_width - 1] =
        lerp1d_scalar(src_row0[src_width - 1], src_row1[src_width - 1]);
    dst_row1[dst_width - 1] =
        lerp1d_scalar(src_row1[src_width - 1], src_row0[src_width - 1]);

    // Middle elements: full 2D interpolation
    for (size_t src_x = 0; src_x + 1 < src_width; ++src_x) {
      size_t dst_x = src_x * 2 + 1;
      const uint8_t tl = src_row0[src_x], tr = src_row0[src_x + 1],
                    bl = src_row1[src_x], br = src_row1[src_x + 1];
      dst_row0[dst_x] = lerp2d_scalar(tl, tr, bl, br);
      dst_row0[dst_x + 1] = lerp2d_scalar(tr, tl, br, bl);
      dst_row1[dst_x] = lerp2d_scalar(bl, tl, br, tr);
      dst_row1[dst_x + 1] = lerp2d_scalar(br, tr, bl, tl);
    }
  };

  // Top row
  if (KLEIDICV_LIKELY(y_begin == 0)) {
    process_edge_row(src, dst);
  }

  // Middle rows
  for (size_t src_y = y_begin; src_y + 1 < y_end; ++src_y) {
    size_t dst_y = src_y * 2 + 1;
    const uint8_t *src_row0 = src + src_stride * src_y;
    const uint8_t *src_row1 = src_row0 + src_stride;
    uint8_t *dst_row0 = dst + dst_stride * dst_y;
    uint8_t *dst_row1 = dst_row0 + dst_stride;

    process_row(src_row0, src_row1, dst_row0, dst_row1);
  }

  // Bottom row
  if (KLEIDICV_LIKELY(y_end == src_height)) {
    process_edge_row(src + src_stride * (src_height - 1),
                     dst + dst_stride * (2 * src_height - 1));
  }

  return KLEIDICV_OK;
}

// ========== 4x4 u8 upscale ==========
// Matches the Neon implementation exactly:
//   Edge rows: dst[0..1] = src[0], dst[w-1..w-2] = src[w-1], middle uses
//              lerp1d with weights (7/8, 5/8, 3/8, 1/8).
//   Interior: 2D interpolation with specific weight tables.

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t kleidicv_resize_4x4_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, uint8_t *dst, size_t dst_stride) {
  void *_ws = std::malloc(src_width * 8);
  if (!_ws) return KLEIDICV_ERROR_ALLOCATION;
  std::free(_ws);

  size_t dst_width = src_width * 4;
  size_t dst_height = src_height * 4;

  auto lerp1d_scalar = [](uint8_t coeff_a, uint8_t a, uint8_t coeff_b,
                          uint8_t b) -> uint8_t {
    return static_cast<uint8_t>((coeff_a * a + coeff_b * b + 4) >> 3);
  };

  auto lerp2d_scalar = [](uint8_t coeff_a, uint8_t a, uint8_t coeff_b,
                          uint8_t b, uint8_t coeff_c, uint8_t c,
                          uint8_t coeff_d, uint8_t d) -> uint8_t {
    return static_cast<uint8_t>(
        (coeff_a * a + coeff_b * b + coeff_c * c + coeff_d * d + 32) >> 6);
  };

  // Handle top or bottom edge
  auto process_edge_row = [src_width, dst_width, lerp1d_scalar](
                              const uint8_t *src_row, uint8_t *dst_row) {
    // Left elements
    dst_row[1] = dst_row[0] = src_row[0];

    // Right elements
    dst_row[dst_width - 1] = dst_row[dst_width - 2] = src_row[src_width - 1];

    // Middle elements
    for (size_t src_x = 0; src_x + 1 < src_width; ++src_x) {
      size_t dst_x = src_x * 4 + 2;
      const uint8_t a = src_row[src_x], b = src_row[src_x + 1];
      dst_row[dst_x + 0] = lerp1d_scalar(7, a, 1, b);
      dst_row[dst_x + 1] = lerp1d_scalar(5, a, 3, b);
      dst_row[dst_x + 2] = lerp1d_scalar(3, a, 5, b);
      dst_row[dst_x + 3] = lerp1d_scalar(1, a, 7, b);
    }
  };

  auto process_row = [src_width, dst_width, lerp1d_scalar, lerp2d_scalar](
                         const uint8_t *src_row0, const uint8_t *src_row1,
                         uint8_t *dst_row0, uint8_t *dst_row1,
                         uint8_t *dst_row2, uint8_t *dst_row3) {
    // Left elements
    const uint8_t s0l = src_row0[0], s1l = src_row1[0];
    dst_row0[0] = dst_row0[1] = lerp1d_scalar(7, s0l, 1, s1l);
    dst_row1[0] = dst_row1[1] = lerp1d_scalar(5, s0l, 3, s1l);
    dst_row2[0] = dst_row2[1] = lerp1d_scalar(3, s0l, 5, s1l);
    dst_row3[0] = dst_row3[1] = lerp1d_scalar(1, s0l, 7, s1l);

    // Right elements
    const uint8_t s0r = src_row0[src_width - 1], s1r = src_row1[src_width - 1];
    const size_t dr0 = dst_width - 2;
    const size_t dr1 = dst_width - 1;
    dst_row0[dr0] = dst_row0[dr1] = lerp1d_scalar(7, s0r, 1, s1r);
    dst_row1[dr0] = dst_row1[dr1] = lerp1d_scalar(5, s0r, 3, s1r);
    dst_row2[dr0] = dst_row2[dr1] = lerp1d_scalar(3, s0r, 5, s1r);
    dst_row3[dr0] = dst_row3[dr1] = lerp1d_scalar(1, s0r, 7, s1r);

    // Middle elements
    for (size_t src_x = 0; src_x + 1 < src_width; ++src_x) {
      size_t dst_x = src_x * 4 + 2;
      const uint8_t a = src_row0[src_x], b = src_row0[src_x + 1],
                    c = src_row1[src_x], d = src_row1[src_x + 1];

      dst_row0[dst_x + 0] = lerp2d_scalar(49, a, 7, b, 7, c, 1, d);
      dst_row0[dst_x + 1] = lerp2d_scalar(35, a, 21, b, 5, c, 3, d);
      dst_row0[dst_x + 2] = lerp2d_scalar(21, a, 35, b, 3, c, 5, d);
      dst_row0[dst_x + 3] = lerp2d_scalar(7, a, 49, b, 1, c, 7, d);
      dst_row1[dst_x + 0] = lerp2d_scalar(35, a, 5, b, 21, c, 3, d);
      dst_row1[dst_x + 1] = lerp2d_scalar(25, a, 15, b, 15, c, 9, d);
      dst_row1[dst_x + 2] = lerp2d_scalar(15, a, 25, b, 9, c, 15, d);
      dst_row1[dst_x + 3] = lerp2d_scalar(5, a, 35, b, 3, c, 21, d);
      dst_row2[dst_x + 0] = lerp2d_scalar(21, a, 3, b, 35, c, 5, d);
      dst_row2[dst_x + 1] = lerp2d_scalar(15, a, 9, b, 25, c, 15, d);
      dst_row2[dst_x + 2] = lerp2d_scalar(9, a, 15, b, 15, c, 25, d);
      dst_row2[dst_x + 3] = lerp2d_scalar(3, a, 21, b, 5, c, 35, d);
      dst_row3[dst_x + 0] = lerp2d_scalar(7, a, 1, b, 49, c, 7, d);
      dst_row3[dst_x + 1] = lerp2d_scalar(5, a, 3, b, 35, c, 21, d);
      dst_row3[dst_x + 2] = lerp2d_scalar(3, a, 5, b, 21, c, 35, d);
      dst_row3[dst_x + 3] = lerp2d_scalar(1, a, 7, b, 7, c, 49, d);
    }
  };

  // Top rows
  if (KLEIDICV_LIKELY(y_begin == 0)) {
    process_edge_row(src, dst);
    std::memcpy(dst + dst_stride, dst, dst_stride);
  }

  // Middle rows
  for (size_t src_y = y_begin; src_y + 1 < y_end; ++src_y) {
    size_t dst_y = src_y * 4 + 2;
    const uint8_t *src_row0 = src + src_stride * src_y;
    const uint8_t *src_row1 = src_row0 + src_stride;
    uint8_t *dst_row0 = dst + dst_stride * dst_y;
    uint8_t *dst_row1 = dst_row0 + dst_stride;
    uint8_t *dst_row2 = dst_row1 + dst_stride;
    uint8_t *dst_row3 = dst_row2 + dst_stride;

    process_row(src_row0, src_row1, dst_row0, dst_row1, dst_row2, dst_row3);
  }

  // Bottom rows
  if (KLEIDICV_LIKELY(y_end == src_height)) {
    process_edge_row(src + src_stride * (src_height - 1),
                     dst + dst_stride * (dst_height - 2));
    std::memcpy(dst + dst_stride * (dst_height - 1),
                dst + dst_stride * (dst_height - 2), dst_stride);
  }

  return KLEIDICV_OK;
}

// ========== f32 resize (2x2, 4x4, 8x8 dispatch) ==========

// --- 2x2 f32 upscale ---
static kleidicv_error_t resize_2x2_f32(
    const float *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, float *dst, size_t dst_stride) {
  size_t dst_width = src_width * 2;
  src_stride /= sizeof(float);
  dst_stride /= sizeof(float);

  auto lerp1d = [](float near, float far) -> float {
    return near * 0.75F + far * 0.25F;
  };

  auto lerp2d = [](float near, float mid_a, float mid_b, float far) -> float {
    return near * 0.5625F + mid_a * 0.1875F + mid_b * 0.1875F + far * 0.0625F;
  };

  // Handle top or bottom edge
  auto process_edge_row = [src_width, dst_width, lerp1d](
                              const float *src_row, float *dst_row) {
    dst_row[0] = src_row[0];
    dst_row[dst_width - 1] = src_row[src_width - 1];

    for (size_t src_x = 0; src_x + 1 < src_width; ++src_x) {
      size_t dst_x = src_x * 2 + 1;
      const float sl = src_row[src_x], sr = src_row[src_x + 1];
      dst_row[dst_x] = lerp1d(sl, sr);
      dst_row[dst_x + 1] = lerp1d(sr, sl);
    }
  };

  auto process_row = [src_width, dst_width, lerp1d, lerp2d](
                         const float *src_row0, const float *src_row1,
                         float *dst_row0, float *dst_row1) {
    dst_row0[0] = lerp1d(src_row0[0], src_row1[0]);
    dst_row1[0] = lerp1d(src_row1[0], src_row0[0]);

    dst_row0[dst_width - 1] =
        lerp1d(src_row0[src_width - 1], src_row1[src_width - 1]);
    dst_row1[dst_width - 1] =
        lerp1d(src_row1[src_width - 1], src_row0[src_width - 1]);

    for (size_t src_x = 0; src_x + 1 < src_width; ++src_x) {
      size_t dst_x = src_x * 2 + 1;
      const float a = src_row0[src_x], b = src_row0[src_x + 1],
                  c = src_row1[src_x], d = src_row1[src_x + 1];
      dst_row0[dst_x] = lerp2d(a, b, c, d);
      dst_row0[dst_x + 1] = lerp2d(b, a, d, c);
      dst_row1[dst_x] = lerp2d(c, a, d, b);
      dst_row1[dst_x + 1] = lerp2d(d, b, c, a);
    }
  };

  if (KLEIDICV_LIKELY(y_begin == 0)) {
    process_edge_row(src, dst);
  }

  for (size_t src_y = y_begin; src_y + 1 < y_end; ++src_y) {
    size_t dst_y = src_y * 2 + 1;
    const float *src_row0 = src + src_stride * src_y;
    const float *src_row1 = src_row0 + src_stride;
    float *dst_row0 = dst + dst_stride * dst_y;
    float *dst_row1 = dst_row0 + dst_stride;

    process_row(src_row0, src_row1, dst_row0, dst_row1);
  }

  if (KLEIDICV_LIKELY(y_end == src_height)) {
    process_edge_row(src + src_stride * (src_height - 1),
                     dst + dst_stride * (src_height * 2 - 1));
  }

  return KLEIDICV_OK;
}

// --- 4x4 f32 upscale ---
static kleidicv_error_t resize_4x4_f32(
    const float *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, float *dst, size_t dst_stride) {
  using T = float;
  size_t dst_height = src_height * 4;
  size_t dst_width = src_width * 4;
  src_stride /= sizeof(T);
  dst_stride /= sizeof(T);

  auto lerp1d = [](T coeff_a, T a, T coeff_b, T b) -> T {
    return coeff_a * a + coeff_b * b;
  };
  auto lerp2d = [](T coeff_a, T a, T coeff_b, T b, T coeff_c, T c,
                    T coeff_d, T d) -> T {
    return coeff_a * a + coeff_b * b + coeff_c * c + coeff_d * d;
  };

  auto process_edge_row = [src_width, dst_width, lerp1d](
                              const T *src_row, T *dst_row) {
    dst_row[1] = dst_row[0] = src_row[0];
    dst_row[dst_width - 1] = dst_row[dst_width - 2] = src_row[src_width - 1];

    for (size_t src_x = 0; src_x + 1 < src_width; ++src_x) {
      size_t dst_x = src_x * 4 + 2;
      const T a = src_row[src_x], b = src_row[src_x + 1];
      dst_row[dst_x + 0] = lerp1d(0.875F, a, 0.125F, b);
      dst_row[dst_x + 1] = lerp1d(0.625F, a, 0.375F, b);
      dst_row[dst_x + 2] = lerp1d(0.375F, a, 0.625F, b);
      dst_row[dst_x + 3] = lerp1d(0.125F, a, 0.875F, b);
    }
  };

  auto process_row = [src_width, dst_width, lerp1d, lerp2d](
                         const T *src_row0, const T *src_row1, T *dst_row0,
                         T *dst_row1, T *dst_row2, T *dst_row3) {
    const T s0l = src_row0[0], s1l = src_row1[0];
    dst_row0[0] = dst_row0[1] = lerp1d(0.875F, s0l, 0.125F, s1l);
    dst_row1[0] = dst_row1[1] = lerp1d(0.625F, s0l, 0.375F, s1l);
    dst_row2[0] = dst_row2[1] = lerp1d(0.375F, s0l, 0.625F, s1l);
    dst_row3[0] = dst_row3[1] = lerp1d(0.125F, s0l, 0.875F, s1l);

    const T s0r = src_row0[src_width - 1], s1r = src_row1[src_width - 1];
    const size_t dr0 = dst_width - 2;
    const size_t dr1 = dst_width - 1;
    dst_row0[dr0] = dst_row0[dr1] = lerp1d(0.875F, s0r, 0.125F, s1r);
    dst_row1[dr0] = dst_row1[dr1] = lerp1d(0.625F, s0r, 0.375F, s1r);
    dst_row2[dr0] = dst_row2[dr1] = lerp1d(0.375F, s0r, 0.625F, s1r);
    dst_row3[dr0] = dst_row3[dr1] = lerp1d(0.125F, s0r, 0.875F, s1r);

    for (size_t src_x = 0; src_x + 1 < src_width; ++src_x) {
      size_t dst_x = src_x * 4 + 2;
      const T a = src_row0[src_x], b = src_row0[src_x + 1],
              c = src_row1[src_x], d = src_row1[src_x + 1];

      dst_row0[dst_x + 0] =
          lerp2d(0.765625F, a, 0.109375F, b, 0.109375F, c, 0.015625F, d);
      dst_row0[dst_x + 1] =
          lerp2d(0.546875F, a, 0.328125F, b, 0.078125F, c, 0.046875F, d);
      dst_row0[dst_x + 2] =
          lerp2d(0.328125F, a, 0.546875F, b, 0.046875F, c, 0.078125F, d);
      dst_row0[dst_x + 3] =
          lerp2d(0.109375F, a, 0.765625F, b, 0.015625F, c, 0.109375F, d);
      dst_row1[dst_x + 0] =
          lerp2d(0.546875F, a, 0.078125F, b, 0.328125F, c, 0.046875F, d);
      dst_row1[dst_x + 1] =
          lerp2d(0.390625F, a, 0.234375F, b, 0.234375F, c, 0.140625F, d);
      dst_row1[dst_x + 2] =
          lerp2d(0.234375F, a, 0.390625F, b, 0.140625F, c, 0.234375F, d);
      dst_row1[dst_x + 3] =
          lerp2d(0.078125F, a, 0.546875F, b, 0.046875F, c, 0.328125F, d);
      dst_row2[dst_x + 0] =
          lerp2d(0.328125F, a, 0.046875F, b, 0.546875F, c, 0.078125F, d);
      dst_row2[dst_x + 1] =
          lerp2d(0.234375F, a, 0.140625F, b, 0.390625F, c, 0.234375F, d);
      dst_row2[dst_x + 2] =
          lerp2d(0.140625F, a, 0.234375F, b, 0.234375F, c, 0.390625F, d);
      dst_row2[dst_x + 3] =
          lerp2d(0.046875F, a, 0.328125F, b, 0.078125F, c, 0.546875F, d);
      dst_row3[dst_x + 0] =
          lerp2d(0.109375F, a, 0.015625F, b, 0.765625F, c, 0.109375F, d);
      dst_row3[dst_x + 1] =
          lerp2d(0.078125F, a, 0.046875F, b, 0.546875F, c, 0.328125F, d);
      dst_row3[dst_x + 2] =
          lerp2d(0.046875F, a, 0.078125F, b, 0.328125F, c, 0.546875F, d);
      dst_row3[dst_x + 3] =
          lerp2d(0.015625F, a, 0.109375F, b, 0.109375F, c, 0.765625F, d);
    }
  };

  if (KLEIDICV_LIKELY(y_begin == 0)) {
    process_edge_row(src, dst);
    std::memcpy(dst + dst_stride, dst, dst_stride * sizeof(T));
  }

  for (size_t src_y = y_begin; src_y + 1 < y_end; ++src_y) {
    size_t dst_y = src_y * 4 + 2;
    const T *src_row0 = src + src_stride * src_y;
    const T *src_row1 = src_row0 + src_stride;
    T *dst_row0 = dst + dst_stride * dst_y;
    T *dst_row1 = dst_row0 + dst_stride;
    T *dst_row2 = dst_row1 + dst_stride;
    T *dst_row3 = dst_row2 + dst_stride;

    process_row(src_row0, src_row1, dst_row0, dst_row1, dst_row2, dst_row3);
  }

  if (KLEIDICV_LIKELY(y_end == src_height)) {
    process_edge_row(src + src_stride * (src_height - 1),
                     dst + dst_stride * (dst_height - 2));
    std::memcpy(dst + dst_stride * (dst_height - 1),
                dst + dst_stride * (dst_height - 2), dst_stride * sizeof(T));
  }

  return KLEIDICV_OK;
}

// --- 8x8 f32 upscale ---
static kleidicv_error_t resize_8x8_f32(
    const float *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, float *dst, size_t dst_stride) {
  size_t dst_width = src_width * 8;
  size_t dst_height = src_height * 8;
  src_stride /= sizeof(float);
  dst_stride /= sizeof(float);

  // Horizontal interpolation coefficients for 8x: positions 0..7 within a
  // source pixel pair. Position i maps to weight (15-2*i)/16 for the left
  // source pixel.
  float coeffs_a[8] = {15 / 16.0f, 13 / 16.0f, 11 / 16.0f, 9 / 16.0f,
                        7 / 16.0f,  5 / 16.0f,  3 / 16.0f,  1 / 16.0f};
  float coeffs_b[8] = {1 / 16.0f, 3 / 16.0f,  5 / 16.0f,  7 / 16.0f,
                        9 / 16.0f, 11 / 16.0f, 13 / 16.0f, 15 / 16.0f};

  // Handle top or bottom edge: 4 rows share the same horizontal values.
  auto process_edge_row = [src_width, dst_width, &coeffs_a, &coeffs_b](
                              const float *src_row, float *dst_base,
                              size_t dst_str) {
    // Left 4 elements
    for (size_t r = 0; r < 4; ++r) {
      float *dr = dst_base + r * dst_str;
      dr[0] = dr[1] = dr[2] = dr[3] = src_row[0];
    }

    // Right 4 elements
    for (size_t r = 0; r < 4; ++r) {
      float *dr = dst_base + r * dst_str + dst_width - 4;
      dr[0] = dr[1] = dr[2] = dr[3] = src_row[src_width - 1];
    }

    // Middle elements
    for (size_t src_x = 0; src_x + 1 < src_width; ++src_x) {
      float a = src_row[src_x], b = src_row[src_x + 1];
      float *base = dst_base + src_x * 8 + 4;
      for (size_t i = 0; i < 8; ++i) {
        float val = coeffs_a[i] * a + coeffs_b[i] * b;
        base[i] = val;
        base[i + dst_str] = val;
        base[i + 2 * dst_str] = val;
        base[i + 3 * dst_str] = val;
      }
    }
  };

  auto process_row = [src_width, &coeffs_a, &coeffs_b](
                         const float *src_row0, const float *src_row1,
                         float *dst_base, size_t dst_str) {
    // Left elements
    float s0 = src_row0[0], s1 = src_row1[0];
    for (size_t i = 0; i < 8; ++i) {
      float vy = static_cast<float>(15 - static_cast<int>(i) * 2) / 16.0f;
      float val = vy * s0 + (1.0f - vy) * s1;
      float *dr = dst_base + i * dst_str;
      dr[0] = dr[1] = dr[2] = dr[3] = val;
    }

    // Middle elements
    float prev_a = s0, prev_c = s1;
    for (size_t src_x = 0; src_x + 1 < src_width; ++src_x) {
      float a = prev_a;
      float b = src_row0[src_x + 1];
      float c = prev_c;
      float d = src_row1[src_x + 1];
      prev_a = b;
      prev_c = d;

      // Compute row 0 (top) and row 7 (bottom) interpolation coefficients.
      // Row 0: vertical weight = 15/16 for src_row0, 1/16 for src_row1
      // Row 7: vertical weight = 1/16 for src_row0, 15/16 for src_row1
      float *dst_ptr = dst_base + src_x * 8 + 4;
      float row0[8], row7[8];
      for (size_t j = 0; j < 8; ++j) {
        float ha = coeffs_a[j], hb = coeffs_b[j];
        // 2D weight = (horizontal weight) * (vertical weight)
        row0[j] = (15.0f / 16.0f) * (ha * a + hb * b) +
                  (1.0f / 16.0f) * (ha * c + hb * d);
        row7[j] = (1.0f / 16.0f) * (ha * a + hb * b) +
                  (15.0f / 16.0f) * (ha * c + hb * d);
      }

      // Store row 0 and row 7 directly
      for (size_t j = 0; j < 8; ++j) {
        dst_ptr[j] = row0[j];
        dst_ptr[7 * dst_str + j] = row7[j];
      }

      // Rows 1-6: linearly interpolate between row 0 and row 7
      for (size_t r = 1; r < 7; ++r) {
        float t = static_cast<float>(r) / 7.0f;
        for (size_t j = 0; j < 8; ++j) {
          dst_ptr[r * dst_str + j] = row0[j] + t * (row7[j] - row0[j]);
        }
      }
    }

    // Right elements
    s0 = prev_a;
    s1 = prev_c;
    float *dst_right = dst_base + (src_width - 1) * 8 + 4;
    for (size_t i = 0; i < 8; ++i) {
      float vy = static_cast<float>(15 - static_cast<int>(i) * 2) / 16.0f;
      float val = vy * s0 + (1.0f - vy) * s1;
      float *dr = dst_right + i * dst_str;
      dr[0] = dr[1] = dr[2] = dr[3] = val;
    }
  };

  // Top rows
  if (KLEIDICV_LIKELY(y_begin == 0)) {
    process_edge_row(src, dst, dst_stride);
  }

  // Middle rows
  for (size_t src_y = y_begin; src_y + 1 < y_end; ++src_y) {
    size_t dst_y = src_y * 8 + 4;
    const float *src_row0 = src + src_stride * src_y;
    const float *src_row1 = src_row0 + src_stride;
    process_row(src_row0, src_row1, dst + dst_stride * dst_y, dst_stride);
  }

  // Bottom rows
  if (KLEIDICV_LIKELY(y_end == src_height)) {
    process_edge_row(src + src_stride * (src_height - 1),
                     dst + dst_stride * (dst_height - 4), dst_stride);
  }

  return KLEIDICV_OK;
}

// --- f32 dispatch ---
kleidicv_error_t kleidicv_resize_linear_stripe_f32(
    const float *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, float *dst, size_t dst_stride,
    size_t dst_width, size_t dst_height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, dst_height);

  if (src_width == 0 || src_height == 0) {
    return KLEIDICV_OK;
  }
  if (src_width * 2 == dst_width && src_height * 2 == dst_height) {
    return resize_2x2_f32(src, src_stride, src_width, src_height, y_begin,
                          y_end, dst, dst_stride);
  }
  if (src_width * 4 == dst_width && src_height * 4 == dst_height) {
    return resize_4x4_f32(src, src_stride, src_width, src_height, y_begin,
                          y_end, dst, dst_stride);
  }
  if (src_width * 8 == dst_width && src_height * 8 == dst_height) {
    return resize_8x8_f32(src, src_stride, src_width, src_height, y_begin,
                          y_end, dst, dst_stride);
  }
  // resize_linear_f32_is_implemented checked the kernel size already.
  assert(!"resize ratio not implemented");
  return KLEIDICV_ERROR_NOT_IMPLEMENTED;
}

}  // namespace kleidicv::neon
