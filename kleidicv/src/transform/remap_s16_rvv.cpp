// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cassert>

#include "kleidicv/rvv.h"
#include "kleidicv/transform/remap.h"

namespace kleidicv::neon {

// Scalar fallback for remap_s16 (integer coordinate map).
template <typename T>
kleidicv_error_t remap_s16(const T *src, size_t src_stride, size_t src_width,
                           size_t src_height, T *dst, size_t dst_stride,
                           size_t dst_width, size_t dst_height, size_t channels,
                           const int16_t *mapxy, size_t mapxy_stride,
                           kleidicv_border_type_t border_type,
                           [[maybe_unused]] const T *border_value) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, dst_height);
  CHECK_POINTER_AND_STRIDE(mapxy, mapxy_stride, dst_height);
  CHECK_IMAGE_SIZE(src_width, src_height);
  CHECK_IMAGE_SIZE(dst_width, dst_height);
  if (border_type == KLEIDICV_BORDER_TYPE_CONSTANT && nullptr == border_value) {
    return KLEIDICV_ERROR_NULL_POINTER;
  }

  if (!remap_s16_is_implemented<T>(src_stride, src_width, src_height, dst_width,
                                   border_type, channels)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  size_t src_elem_stride = src_stride / sizeof(T);

  for (size_t y = 0; y < dst_height; ++y) {
    const int16_t *map_row = row_ptr(mapxy, mapxy_stride, y);
    T *dst_row = row_ptr(dst, dst_stride, y);

    for (size_t x = 0; x < dst_width; ++x) {
      int16_t mx = map_row[x * 2];
      int16_t my = map_row[x * 2 + 1];

      if (border_type == KLEIDICV_BORDER_TYPE_REPLICATE) {
        mx = static_cast<int16_t>(
            std::clamp(static_cast<int>(mx), 0,
                       static_cast<int>(src_width) - 1));
        my = static_cast<int16_t>(
            std::clamp(static_cast<int>(my), 0,
                       static_cast<int>(src_height) - 1));
        dst_row[x] = src[static_cast<size_t>(my) * src_elem_stride +
                         static_cast<size_t>(mx)];
      } else {
        // Constant border
        if (mx < 0 || mx >= static_cast<int16_t>(src_width) || my < 0 ||
            my >= static_cast<int16_t>(src_height)) {
          dst_row[x] = *border_value;
        } else {
          dst_row[x] = src[static_cast<size_t>(my) * src_elem_stride +
                           static_cast<size_t>(mx)];
        }
      }
    }
  }
  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16(type)                          \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t remap_s16<type>(          \
      const type *src, size_t src_stride, size_t src_width, size_t src_height, \
      type *dst, size_t dst_stride, size_t dst_width, size_t dst_height,       \
      size_t channels, const int16_t *mapxy, size_t mapxy_stride,              \
      kleidicv_border_type_t border_type, const type *border_value)

KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE_REMAP_S16(uint16_t);

}  // namespace kleidicv::neon
