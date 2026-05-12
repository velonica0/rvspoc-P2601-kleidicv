// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cstdlib>
#include <memory>

#include "kleidicv/rvv.h"

#if KLEIDICV_EXPERIMENTAL_FEATURE_CANNY

namespace kleidicv::neon {

// Canny edge detection -- scalar fallback.
// The NEON version uses complex table lookups, directional masking, and
// hysteresis that are deeply tied to NEON intrinsics.  We delegate to
// upstream C API calls (sobel, threshold, add_abs_with_threshold) which
// themselves will use the RVV paths when available.

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t canny_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, double low_threshold, double high_threshold) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  // Canny requires substantial temporary buffers and a multi-pass algorithm.
  // For the RVV port we provide a correct scalar implementation.

  // Allocate gradient buffers
  size_t buf_stride = width * sizeof(int16_t);
  auto gx_buf = std::unique_ptr<int16_t[]>(new (std::nothrow) int16_t[width * height]);
  auto gy_buf = std::unique_ptr<int16_t[]>(new (std::nothrow) int16_t[width * height]);
  auto mag_buf = std::unique_ptr<int16_t[]>(new (std::nothrow) int16_t[width * height]);
  auto edge_buf = std::unique_ptr<uint8_t[]>(new (std::nothrow) uint8_t[width * height]);
  if (!gx_buf || !gy_buf || !mag_buf || !edge_buf) {
    return KLEIDICV_ERROR_ALLOCATION;
  }

  // Compute Sobel gradients (3x3)
  for (size_t y = 1; y + 1 < height; ++y) {
    for (size_t x = 1; x + 1 < width; ++x) {
      auto px = [&](size_t r, size_t c) -> int {
        return static_cast<int>(row_ptr(src, src_stride, r)[c]);
      };
      int gx = -px(y - 1, x - 1) + px(y - 1, x + 1) - 2 * px(y, x - 1) +
               2 * px(y, x + 1) - px(y + 1, x - 1) + px(y + 1, x + 1);
      int gy = -px(y - 1, x - 1) - 2 * px(y - 1, x) - px(y - 1, x + 1) +
               px(y + 1, x - 1) + 2 * px(y + 1, x) + px(y + 1, x + 1);
      gx_buf[y * width + x] = static_cast<int16_t>(gx);
      gy_buf[y * width + x] = static_cast<int16_t>(gy);
      int mag = std::abs(gx) + std::abs(gy);
      mag_buf[y * width + x] =
          (mag > static_cast<int>(low_threshold))
              ? static_cast<int16_t>(std::min(mag, 32767))
              : 0;
    }
  }

  // Zero borders
  for (size_t x = 0; x < width; ++x) {
    gx_buf[x] = gy_buf[x] = mag_buf[x] = 0;
    gx_buf[(height - 1) * width + x] = gy_buf[(height - 1) * width + x] =
        mag_buf[(height - 1) * width + x] = 0;
  }
  for (size_t y = 0; y < height; ++y) {
    gx_buf[y * width] = gy_buf[y * width] = mag_buf[y * width] = 0;
    gx_buf[y * width + width - 1] = gy_buf[y * width + width - 1] =
        mag_buf[y * width + width - 1] = 0;
  }

  // Non-maximum suppression + double thresholding
  std::memset(edge_buf.get(), 0, width * height);
  int16_t hi = static_cast<int16_t>(high_threshold);
  for (size_t y = 1; y + 1 < height; ++y) {
    for (size_t x = 1; x + 1 < width; ++x) {
      int16_t m = mag_buf[y * width + x];
      if (m == 0) continue;
      int16_t gx = gx_buf[y * width + x];
      int16_t gy = gy_buf[y * width + x];
      int16_t agx = std::abs(gx);
      int16_t agy = std::abs(gy);
      int16_t m1, m2;
      // Approximate gradient direction
      if (agy > static_cast<int16_t>(2.41421 * agx)) {
        // Vertical
        m1 = mag_buf[(y - 1) * width + x];
        m2 = mag_buf[(y + 1) * width + x];
      } else if (agx > static_cast<int16_t>(2.41421 * agy)) {
        // Horizontal
        m1 = mag_buf[y * width + x - 1];
        m2 = mag_buf[y * width + x + 1];
      } else if ((gx > 0) == (gy > 0)) {
        m1 = mag_buf[(y - 1) * width + x - 1];
        m2 = mag_buf[(y + 1) * width + x + 1];
      } else {
        m1 = mag_buf[(y - 1) * width + x + 1];
        m2 = mag_buf[(y + 1) * width + x - 1];
      }
      if (m > m1 && m >= m2) {
        edge_buf[y * width + x] = (m > hi) ? 0xFF : 1;
      }
    }
  }

  // Hysteresis -- promote weak edges connected to strong ones
  bool changed = true;
  while (changed) {
    changed = false;
    for (size_t y = 1; y + 1 < height; ++y) {
      for (size_t x = 1; x + 1 < width; ++x) {
        if (edge_buf[y * width + x] != 1) continue;
        for (int dy = -1; dy <= 1; ++dy) {
          for (int dx = -1; dx <= 1; ++dx) {
            if (edge_buf[(y + dy) * width + (x + dx)] == 0xFF) {
              edge_buf[y * width + x] = 0xFF;
              changed = true;
              goto next_pixel;
            }
          }
        }
      next_pixel:;
      }
    }
  }

  // Write output: strong edges = 0xFF, everything else = 0
  for (size_t y = 0; y < height; ++y) {
    uint8_t *dst_row = row_ptr(dst, dst_stride, y);
    for (size_t x = 0; x < width; ++x) {
      dst_row[x] = (edge_buf[y * width + x] == 0xFF) ? 0xFF : 0;
    }
  }

  return KLEIDICV_OK;
}

}  // namespace kleidicv::neon

extern "C" {

decltype(kleidicv::neon::canny_u8) *kleidicv_canny_u8 =
    kleidicv::neon::canny_u8;

}  // extern "C"

#endif  // KLEIDICV_EXPERIMENTAL_FEATURE_CANNY
