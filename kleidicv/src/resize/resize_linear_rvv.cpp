// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

// Scalar bilinear resize 2x2 upscale for u8.
KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t kleidicv_resize_2x2_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, uint8_t *dst, size_t dst_stride) {
  size_t dst_width = src_width * 2;

  auto lerp1d = [](uint8_t near, uint8_t far) -> uint8_t {
    return static_cast<uint8_t>((near * 3 + far + 2) >> 2);
  };

  for (size_t sy = y_begin; sy < y_end; ++sy) {
    const uint8_t *src_row = row_ptr(src, src_stride, sy);
    const uint8_t *src_row_next =
        (sy + 1 < src_height)
            ? row_ptr(src, src_stride, sy + 1)
            : src_row;
    uint8_t *dst_row0 = row_ptr(dst, dst_stride, sy * 2);
    uint8_t *dst_row1 = row_ptr(dst, dst_stride, sy * 2 + 1);

    for (size_t sx = 0; sx < src_width; ++sx) {
      uint8_t tl = src_row[sx];
      uint8_t tr = (sx + 1 < src_width) ? src_row[sx + 1] : tl;
      uint8_t bl = src_row_next[sx];
      uint8_t br = (sx + 1 < src_width) ? src_row_next[sx + 1] : bl;

      dst_row0[sx * 2] = lerp1d(tl, tr);
      dst_row0[sx * 2 + 1] = lerp1d(tr, tl);
      dst_row1[sx * 2] = lerp1d(lerp1d(tl, bl), lerp1d(tr, br));
      dst_row1[sx * 2 + 1] = lerp1d(lerp1d(tr, br), lerp1d(tl, bl));
    }
  }
  return KLEIDICV_OK;
}

// Scalar bilinear resize 4x4 upscale for u8.
KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t kleidicv_resize_4x4_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, uint8_t *dst, size_t dst_stride) {
  // Scalar bilinear interpolation for 4x upscale
  size_t dst_width = src_width * 4;
  size_t dst_height = src_height * 4;

  for (size_t sy = y_begin; sy < y_end; ++sy) {
    for (int sub_y = 0; sub_y < 4; ++sub_y) {
      size_t dy = sy * 4 + sub_y;
      if (dy >= dst_height) break;
      uint8_t *dst_row = row_ptr(dst, dst_stride, dy);
      float fy = (static_cast<float>(dy) + 0.5f) / 4.0f - 0.5f;
      int iy = static_cast<int>(std::floor(fy));
      float yfrac = fy - static_cast<float>(iy);
      iy = std::clamp(iy, 0, static_cast<int>(src_height) - 1);
      int iy1 = std::min(iy + 1, static_cast<int>(src_height) - 1);
      const uint8_t *row0 = row_ptr(src, src_stride, static_cast<size_t>(iy));
      const uint8_t *row1 = row_ptr(src, src_stride, static_cast<size_t>(iy1));

      for (size_t dx = 0; dx < dst_width; ++dx) {
        float fx = (static_cast<float>(dx) + 0.5f) / 4.0f - 0.5f;
        int ix = static_cast<int>(std::floor(fx));
        float xfrac = fx - static_cast<float>(ix);
        ix = std::clamp(ix, 0, static_cast<int>(src_width) - 1);
        int ix1 = std::min(ix + 1, static_cast<int>(src_width) - 1);

        float a = static_cast<float>(row0[ix]);
        float b = static_cast<float>(row0[ix1]);
        float c = static_cast<float>(row1[ix]);
        float d = static_cast<float>(row1[ix1]);
        float val = a * (1 - xfrac) * (1 - yfrac) +
                    b * xfrac * (1 - yfrac) + c * (1 - xfrac) * yfrac +
                    d * xfrac * yfrac;
        dst_row[dx] =
            static_cast<uint8_t>(std::clamp(val + 0.5f, 0.0f, 255.0f));
      }
    }
  }
  return KLEIDICV_OK;
}

// Scalar bilinear resize for f32.
kleidicv_error_t kleidicv_resize_linear_stripe_f32(
    const float *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, float *dst, size_t dst_stride,
    size_t dst_width, size_t dst_height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, dst_height);

  if (src_width == 0 || src_height == 0) {
    return KLEIDICV_OK;
  }

  for (size_t dy = y_begin; dy < y_end; ++dy) {
    float *dst_row = row_ptr(dst, dst_stride, dy);
    float fy =
        (static_cast<float>(dy) + 0.5f) *
            static_cast<float>(src_height) / static_cast<float>(dst_height) -
        0.5f;
    int iy = static_cast<int>(std::floor(fy));
    float yfrac = fy - static_cast<float>(iy);
    iy = std::clamp(iy, 0, static_cast<int>(src_height) - 1);
    int iy1 = std::min(iy + 1, static_cast<int>(src_height) - 1);
    const float *row0 = row_ptr(src, src_stride, static_cast<size_t>(iy));
    const float *row1 = row_ptr(src, src_stride, static_cast<size_t>(iy1));

    for (size_t dx = 0; dx < dst_width; ++dx) {
      float fx =
          (static_cast<float>(dx) + 0.5f) *
              static_cast<float>(src_width) / static_cast<float>(dst_width) -
          0.5f;
      int ix = static_cast<int>(std::floor(fx));
      float xfrac = fx - static_cast<float>(ix);
      ix = std::clamp(ix, 0, static_cast<int>(src_width) - 1);
      int ix1 = std::min(ix + 1, static_cast<int>(src_width) - 1);

      float a = row0[ix];
      float b = row0[ix1];
      float c = row1[ix];
      float d = row1[ix1];
      dst_row[dx] = a * (1 - xfrac) * (1 - yfrac) +
                    b * xfrac * (1 - yfrac) + c * (1 - xfrac) * yfrac +
                    d * xfrac * yfrac;
    }
  }
  return KLEIDICV_OK;
}

}  // namespace kleidicv::neon
