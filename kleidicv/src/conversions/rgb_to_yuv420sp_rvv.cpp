// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <utility>

#include "kleidicv/conversions/rgb_to_yuv.h"
#include "kleidicv/kleidicv.h"

#include "kleidicv/rvv.h"

#include "yuv42x_coefficients.h"

namespace kleidicv::neon {

template <bool kAlpha, bool RGB>
static kleidicv_error_t rgb_to_yuv420sp_scalar(
    const uint8_t *src, size_t src_stride, uint8_t *y_dst, size_t y_stride,
    uint8_t *uv_dst, size_t uv_stride, size_t width, size_t height,
    bool v_first, size_t begin, size_t end) {
  constexpr size_t r_idx = RGB ? 0 : 2;
  constexpr size_t g_idx = 1;
  constexpr size_t b_idx = RGB ? 2 : 0;
  constexpr size_t scn = kAlpha ? 4 : 3;

  const int kShifted16 = (16 << kWeightScale);
  const int kHalfShift = (1 << (kWeightScale - 1));
  const int kShifted128 = (128 << kWeightScale);

  size_t row_begin = begin * 2;
  size_t row_end = std::min<size_t>(height, end * 2);

  for (size_t h = row_begin; h < row_end; ++h) {
    const uint8_t *src_row = src + src_stride * h;
    uint8_t *y_row = y_dst + y_stride * h;

    bool evenRow = (h & 1) == 0;
    uint8_t *uv_row = nullptr;
    if (evenRow) {
      uv_row = uv_dst + uv_stride * (h / 2);
    }

    for (size_t x = 0; x < width; ++x) {
      uint8_t r = src_row[x * scn + r_idx];
      uint8_t g = src_row[x * scn + g_idx];
      uint8_t b = src_row[x * scn + b_idx];

      int yy = kRYWeight * r + kGYWeight * g + kBYWeight * b + kHalfShift +
               kShifted16;
      y_row[x] = static_cast<uint8_t>(std::clamp(yy >> kWeightScale, 0, 255));

      bool evenCol = (x & 1) == 0;
      if (evenRow && evenCol) {
        int uu = kRUWeight * r + kGUWeight * g + kBUWeight * b + kHalfShift +
                 kShifted128;
        int vv = kBUWeight * r + kGVWeight * g + kBVWeight * b + kHalfShift +
                 kShifted128;

        uint8_t u_val =
            static_cast<uint8_t>(std::clamp(uu >> kWeightScale, 0, 255));
        uint8_t v_val =
            static_cast<uint8_t>(std::clamp(vv >> kWeightScale, 0, 255));

        if (v_first) std::swap(u_val, v_val);

        uv_row[x] = u_val;
        uv_row[x + 1] = v_val;
      }
    }
  }

  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgb_to_yuv420sp_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *y_dst, size_t y_stride,
    uint8_t *uv_dst, size_t uv_stride, size_t width, size_t height,
    kleidicv_color_conversion_t color_format, size_t begin, size_t end) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(y_dst, y_stride, height);
  CHECK_POINTER_AND_STRIDE(uv_dst, uv_stride, (height + 1) / 2);
  CHECK_IMAGE_SIZE(width, height);

  switch (color_format) {
    case KLEIDICV_BGR_TO_NV21:
      return rgb_to_yuv420sp_scalar<false, false>(
          src, src_stride, y_dst, y_stride, uv_dst, uv_stride, width, height,
          true, begin, end);
    case KLEIDICV_RGB_TO_NV21:
      return rgb_to_yuv420sp_scalar<false, true>(
          src, src_stride, y_dst, y_stride, uv_dst, uv_stride, width, height,
          true, begin, end);
    case KLEIDICV_BGRA_TO_NV21:
      return rgb_to_yuv420sp_scalar<true, false>(
          src, src_stride, y_dst, y_stride, uv_dst, uv_stride, width, height,
          true, begin, end);
    case KLEIDICV_RGBA_TO_NV21:
      return rgb_to_yuv420sp_scalar<true, true>(
          src, src_stride, y_dst, y_stride, uv_dst, uv_stride, width, height,
          true, begin, end);
    case KLEIDICV_BGR_TO_NV12:
      return rgb_to_yuv420sp_scalar<false, false>(
          src, src_stride, y_dst, y_stride, uv_dst, uv_stride, width, height,
          false, begin, end);
    case KLEIDICV_RGB_TO_NV12:
      return rgb_to_yuv420sp_scalar<false, true>(
          src, src_stride, y_dst, y_stride, uv_dst, uv_stride, width, height,
          false, begin, end);
    case KLEIDICV_BGRA_TO_NV12:
      return rgb_to_yuv420sp_scalar<true, false>(
          src, src_stride, y_dst, y_stride, uv_dst, uv_stride, width, height,
          false, begin, end);
    case KLEIDICV_RGBA_TO_NV12:
      return rgb_to_yuv420sp_scalar<true, true>(
          src, src_stride, y_dst, y_stride, uv_dst, uv_stride, width, height,
          false, begin, end);
    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
}

}  // namespace kleidicv::neon
