// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cassert>

#include "kleidicv/rvv.h"
#include "kleidicv/transform/remap.h"

namespace kleidicv::neon {

// Scalar fallback for remap_s16point5 (fixed-point subpixel remap).
template <typename T>
kleidicv_error_t remap_s16point5(
    const T *src, size_t src_stride, size_t src_width, size_t src_height,
    T *dst, size_t dst_stride, size_t dst_width, size_t dst_height,
    size_t channels, const int16_t *mapxy, size_t mapxy_stride,
    const uint16_t *mapfrac, size_t mapfrac_stride,
    [[maybe_unused]] kleidicv_border_type_t border_type,
    [[maybe_unused]] const T *border_value) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, dst_height);
  CHECK_POINTER_AND_STRIDE(mapxy, mapxy_stride, dst_height);
  CHECK_POINTER_AND_STRIDE(mapfrac, mapfrac_stride, dst_height);
  CHECK_IMAGE_SIZE(src_width, src_height);
  CHECK_IMAGE_SIZE(dst_width, dst_height);
  if (border_type == KLEIDICV_BORDER_TYPE_CONSTANT && nullptr == border_value) {
    return KLEIDICV_ERROR_NULL_POINTER;
  }

  if (!remap_s16point5_is_implemented<T>(src_stride, src_width, src_height,
                                         dst_width, border_type, channels)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  auto clamp_coord = [](int v, int max_val) -> int {
    return std::clamp(v, 0, max_val);
  };

  auto get_pixel = [&](int ix, int iy, size_t ch) -> uint32_t {
    if (border_type == KLEIDICV_BORDER_TYPE_REPLICATE) {
      ix = clamp_coord(ix, static_cast<int>(src_width) - 1);
      iy = clamp_coord(iy, static_cast<int>(src_height) - 1);
    } else if (ix < 0 || ix >= static_cast<int>(src_width) || iy < 0 ||
               iy >= static_cast<int>(src_height)) {
      return static_cast<uint32_t>(border_value[ch]);
    }
    return static_cast<uint32_t>(
        row_ptr(src, src_stride,
                static_cast<size_t>(iy))[static_cast<size_t>(ix) * channels +
                                          ch]);
  };

  for (size_t y = 0; y < dst_height; ++y) {
    const int16_t *map_row = row_ptr(mapxy, mapxy_stride, y);
    const uint16_t *frac_row = row_ptr(mapfrac, mapfrac_stride, y);
    T *dst_row = row_ptr(dst, dst_stride, y);

    for (size_t x = 0; x < dst_width; ++x) {
      int16_t mx = map_row[x * 2];
      int16_t my = map_row[x * 2 + 1];
      uint16_t frac = frac_row[x];
      uint16_t xfrac = frac & (REMAP16POINT5_FRAC_MAX - 1);
      uint16_t yfrac =
          (frac >> REMAP16POINT5_FRAC_BITS) & (REMAP16POINT5_FRAC_MAX - 1);

      if (border_type == KLEIDICV_BORDER_TYPE_REPLICATE) {
        if (mx < 0) xfrac = 0;
        if (my < 0) yfrac = 0;
      }

      uint16_t nxfrac = REMAP16POINT5_FRAC_MAX - xfrac;
      uint16_t nyfrac = REMAP16POINT5_FRAC_MAX - yfrac;

      for (size_t ch = 0; ch < channels; ++ch) {
        uint32_t a = get_pixel(mx, my, ch);
        uint32_t b = get_pixel(mx + 1, my, ch);
        uint32_t c = get_pixel(mx, my + 1, ch);
        uint32_t d = get_pixel(mx + 1, my + 1, ch);

        uint32_t line0 = nxfrac * a + xfrac * b;
        uint32_t line1 = nxfrac * c + xfrac * d;
        uint32_t val = (line0 * nyfrac + line1 * yfrac +
                        REMAP16POINT5_FRAC_MAX_SQUARE / 2) >>
                       (2 * REMAP16POINT5_FRAC_BITS);
        dst_row[x * channels + ch] = static_cast<T>(val);
      }
    }
  }
  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16Point5(type)                    \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t remap_s16point5<type>(    \
      const type *src, size_t src_stride, size_t src_width, size_t src_height, \
      type *dst, size_t dst_stride, size_t dst_width, size_t dst_height,       \
      size_t channels, const int16_t *mapxy, size_t mapxy_stride,              \
      const uint16_t *mapfrac, size_t mapfrac_stride,                          \
      kleidicv_border_type_t border_type, const type *border_value)

KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16Point5(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16Point5(uint16_t);

}  // namespace kleidicv::neon
