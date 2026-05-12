// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cassert>
#include <cmath>

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
        int ix = static_cast<int>(std::round(xt));
        int iy = static_cast<int>(std::round(yt));
        dst_row[x] = get_pixel(ix, iy);
      } else {
        int ix = static_cast<int>(std::floor(xt));
        int iy = static_cast<int>(std::floor(yt));
        float xfrac = xt - static_cast<float>(ix);
        float yfrac = yt - static_cast<float>(iy);
        float a = static_cast<float>(get_pixel(ix, iy));
        float b = static_cast<float>(get_pixel(ix + 1, iy));
        float c = static_cast<float>(get_pixel(ix, iy + 1));
        float d = static_cast<float>(get_pixel(ix + 1, iy + 1));
        float val = a * (1 - xfrac) * (1 - yfrac) +
                    b * xfrac * (1 - yfrac) + c * (1 - xfrac) * yfrac +
                    d * xfrac * yfrac;
        dst_row[x] = static_cast<T>(std::clamp(val + 0.5f, 0.0f, 255.0f));
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
