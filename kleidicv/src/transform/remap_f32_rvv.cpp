// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cassert>
#include <cmath>

#include "kleidicv/rvv.h"
#include "kleidicv/transform/remap.h"

namespace kleidicv::neon {

// Scalar fallback for remap_f32.
template <typename T>
kleidicv_error_t remap_f32(const T *src, size_t src_stride, size_t src_width,
                           size_t src_height, T *dst, size_t dst_stride,
                           size_t dst_width, size_t dst_height, size_t channels,
                           const float *mapx, size_t mapx_stride,
                           const float *mapy, size_t mapy_stride,
                           kleidicv_interpolation_type_t interpolation,
                           kleidicv_border_type_t border_type,
                           const T *border_value) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, dst_height);
  CHECK_POINTER_AND_STRIDE(mapx, mapx_stride, dst_height);
  CHECK_POINTER_AND_STRIDE(mapy, mapy_stride, dst_height);
  CHECK_IMAGE_SIZE(src_width, src_height);
  CHECK_IMAGE_SIZE(dst_width, dst_height);
  if (border_type == KLEIDICV_BORDER_TYPE_CONSTANT && nullptr == border_value) {
    return KLEIDICV_ERROR_NULL_POINTER;
  }

  if (!remap_f32_is_implemented<T>(src_stride, src_width, src_height, dst_width,
                                   dst_height, border_type, channels,
                                   interpolation)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  auto get_pixel = [&](int ix, int iy, size_t ch) -> T {
    if (ix < 0 || ix >= static_cast<int>(src_width) || iy < 0 ||
        iy >= static_cast<int>(src_height)) {
      if (border_type == KLEIDICV_BORDER_TYPE_REPLICATE) {
        ix = std::clamp(ix, 0, static_cast<int>(src_width) - 1);
        iy = std::clamp(iy, 0, static_cast<int>(src_height) - 1);
      } else {
        return border_value[ch];
      }
    }
    return row_ptr(src, src_stride, static_cast<size_t>(iy))
        [static_cast<size_t>(ix) * channels + ch];
  };

  for (size_t y = 0; y < dst_height; ++y) {
    const float *mx = row_ptr(mapx, mapx_stride, y);
    const float *my = row_ptr(mapy, mapy_stride, y);
    T *dst_row = row_ptr(dst, dst_stride, y);

    for (size_t x = 0; x < dst_width; ++x) {
      float fx = mx[x];
      float fy = my[x];

      if (interpolation == KLEIDICV_INTERPOLATION_NEAREST) {
        int ix = static_cast<int>(std::round(fx));
        int iy = static_cast<int>(std::round(fy));
        for (size_t ch = 0; ch < channels; ++ch) {
          dst_row[x * channels + ch] = get_pixel(ix, iy, ch);
        }
      } else {
        // Bilinear
        int ix = static_cast<int>(std::floor(fx));
        int iy = static_cast<int>(std::floor(fy));
        float xfrac = fx - static_cast<float>(ix);
        float yfrac = fy - static_cast<float>(iy);
        for (size_t ch = 0; ch < channels; ++ch) {
          float a = static_cast<float>(get_pixel(ix, iy, ch));
          float b = static_cast<float>(get_pixel(ix + 1, iy, ch));
          float c = static_cast<float>(get_pixel(ix, iy + 1, ch));
          float d = static_cast<float>(get_pixel(ix + 1, iy + 1, ch));
          float val = a * (1 - xfrac) * (1 - yfrac) +
                      b * xfrac * (1 - yfrac) + c * (1 - xfrac) * yfrac +
                      d * xfrac * yfrac;
          dst_row[x * channels + ch] =
              static_cast<T>(std::clamp(val + 0.5f, 0.0f, 255.0f));
        }
      }
    }
  }

  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_F32(type)                          \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t remap_f32<type>(          \
      const type *src, size_t src_stride, size_t src_width, size_t src_height, \
      type *dst, size_t dst_stride, size_t dst_width, size_t dst_height,       \
      size_t channels, const float *mapx, size_t mapx_stride,                  \
      const float *mapy, size_t mapy_stride,                                   \
      kleidicv_interpolation_type_t interpolation,                             \
      kleidicv_border_type_t border_type, const type *border_value)

KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_F32(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_F32(uint16_t);

}  // namespace kleidicv::neon
