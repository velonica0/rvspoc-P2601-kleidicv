// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cassert>
#include <climits>
#include <cmath>

#include "kleidicv/kleidicv.h"
#include "kleidicv/rvv.h"
#include "kleidicv/transform/warp_perspective.h"

namespace kleidicv::neon {

// Scalar fallback for warp_perspective_stripe.
template <typename T>
kleidicv_error_t warp_perspective_stripe(
    const T *src, size_t src_stride, size_t src_width, size_t src_height,
    T *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    size_t y_begin, size_t y_end, const float transformation[9],
    size_t channels, kleidicv_interpolation_type_t interpolation,
    kleidicv_border_type_t border_type, const T *border_value) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, dst_height);
  CHECK_POINTERS(transformation);
  CHECK_IMAGE_SIZE(src_width, src_height);
  CHECK_IMAGE_SIZE(dst_width, dst_height);

  if (border_type == KLEIDICV_BORDER_TYPE_CONSTANT && nullptr == border_value) {
    return KLEIDICV_ERROR_NULL_POINTER;
  }

  if (src_width >= (1ULL << 24) || src_height >= (1ULL << 24) ||
      dst_width >= (1ULL << 24) || dst_height >= (1ULL << 24) ||
      src_stride >= (1ULL << 32) || src_width == 0 || src_height == 0) {
    return KLEIDICV_ERROR_RANGE;
  }

  auto get_pixel = [&](int ix, int iy) -> T {
    if (ix < 0 || ix >= static_cast<int>(src_width) || iy < 0 ||
        iy >= static_cast<int>(src_height)) {
      if (border_type == KLEIDICV_BORDER_TYPE_REPLICATE) {
        ix = std::clamp(ix, 0, static_cast<int>(src_width) - 1);
        iy = std::clamp(iy, 0, static_cast<int>(src_height) - 1);
      } else {
        return border_value ? *border_value : T{0};
      }
    }
    return row_ptr(src, src_stride, static_cast<size_t>(iy))
        [static_cast<size_t>(ix)];
  };

  const float *T_ = transformation;
  for (size_t y = y_begin; y < y_end; ++y) {
    T *dst_row = row_ptr(dst, dst_stride, y);
    float dy = static_cast<float>(y);
    for (size_t x = 0; x < dst_width; ++x) {
      float dx = static_cast<float>(x);
      float tw = T_[6] * dx + T_[7] * dy + T_[8];
      float iw = 1.0f / tw;
      float xt = (T_[0] * dx + T_[1] * dy + T_[2]) * iw;
      float yt = (T_[3] * dx + T_[4] * dy + T_[5]) * iw;

      if (interpolation == KLEIDICV_INTERPOLATION_NEAREST) {
        // Clamp to [INT_MIN, INT_MAX] (via double) to avoid UB on inf/NaN.
        // Any out-of-bounds coordinate hits the border path in get_pixel.
        double rx = std::round(static_cast<double>(xt));
        double ry = std::round(static_cast<double>(yt));
        int ix = static_cast<int>(std::max(static_cast<double>(INT_MIN),
                                           std::min(rx, static_cast<double>(INT_MAX))));
        int iy = static_cast<int>(std::max(static_cast<double>(INT_MIN),
                                           std::min(ry, static_cast<double>(INT_MAX))));
        dst_row[x] = get_pixel(ix, iy);
      } else {
        // Clamp to [INT_MIN, INT_MAX] (via double) to avoid UB on inf/NaN.
        double xt_d = static_cast<double>(xt);
        double yt_d = static_cast<double>(yt);
        double xt_floor = std::floor(xt_d);
        double yt_floor = std::floor(yt_d);
        int ix = static_cast<int>(std::max(static_cast<double>(INT_MIN),
                                           std::min(xt_floor, static_cast<double>(INT_MAX))));
        int iy = static_cast<int>(std::max(static_cast<double>(INT_MIN),
                                           std::min(yt_floor, static_cast<double>(INT_MAX))));
        double xfrac = xt_d - xt_floor;
        double yfrac = yt_d - yt_floor;
        double a = static_cast<double>(get_pixel(ix, iy));
        double b = static_cast<double>(get_pixel(ix + 1, iy));
        double c = static_cast<double>(get_pixel(ix, iy + 1));
        double d = static_cast<double>(get_pixel(ix + 1, iy + 1));
        // Match the reference: line-by-line bilinear in double, then lround.
        double line1 = (b - a) * xfrac + a;
        double line2 = (d - c) * xfrac + c;
        double val = (line2 - line1) * yfrac + line1;
        dst_row[x] = static_cast<T>(std::lround(val));
      }
    }
  }
  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_WARP_PERSPECTIVE(type)                            \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t                           \
  warp_perspective_stripe<type>(                                               \
      const type *src, size_t src_stride, size_t src_width, size_t src_height, \
      type *dst, size_t dst_stride, size_t dst_width, size_t dst_height,       \
      size_t y_begin, size_t y_end, const float transformation[9],             \
      size_t channels, kleidicv_interpolation_type_t interpolation,            \
      kleidicv_border_type_t border_type, const type *border_value)

KLEIDICV_INSTANTIATE_WARP_PERSPECTIVE(uint8_t);

}  // namespace kleidicv::neon
