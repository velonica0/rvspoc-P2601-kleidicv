// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstring>
#include <vector>

#include "kleidicv/rvv.h"
#include "kleidicv/workspace/border_types.h"
#include "median_blur_border_handling.h"

namespace kleidicv::neon {

// Scalar median blur using sorting-network style implementation.
// For each output pixel, gather the kernel window, then use
// std::nth_element to find the median efficiently.

template <typename T>
kleidicv_error_t median_blur_sorting_network_stripe(
    const T *src, size_t src_stride, T *dst, size_t dst_stride, size_t width,
    size_t height, size_t y_begin, size_t y_end, size_t channels,
    size_t kernel_width, [[maybe_unused]] size_t kernel_height,
    FixedBorderType border_type) {
  const size_t ksize = kernel_width;
  const ptrdiff_t half = static_cast<ptrdiff_t>(ksize / 2);
  const size_t window_area = ksize * ksize;
  const size_t median_idx = window_area / 2;

  // Temporary buffer for the kernel window values.
  std::vector<T> window(window_area);

  Rows<const T> src_rows{src, src_stride, channels};
  Rows<T> dst_rows{dst, dst_stride, channels};

  for (size_t y = y_begin; y < y_end; ++y) {
    for (size_t x = 0; x < width; ++x) {
      for (size_t c = 0; c < channels; ++c) {
        size_t idx = 0;
        for (ptrdiff_t ky = -half; ky <= half; ++ky) {
          ptrdiff_t sy = get_physical_index(
              static_cast<size_t>(static_cast<ptrdiff_t>(y) + ky), height,
              border_type);
          for (ptrdiff_t kx = -half; kx <= half; ++kx) {
            ptrdiff_t sx = get_physical_index(
                static_cast<size_t>(static_cast<ptrdiff_t>(x) + kx), width,
                border_type);
            window[idx++] = src_rows.at(sy, sx)[static_cast<ptrdiff_t>(c)];
          }
        }
        std::nth_element(window.begin(), window.begin() + median_idx,
                         window.end());
        dst_rows.at(static_cast<ptrdiff_t>(y),
                    static_cast<ptrdiff_t>(x))[static_cast<ptrdiff_t>(c)] =
            window[median_idx];
      }
    }
  }

  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                                   \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t                          \
  median_blur_sorting_network_stripe<type>(                                   \
      const type *src, size_t src_stride, type *dst, size_t dst_stride,       \
      size_t width, size_t height, size_t y_begin, size_t y_end,              \
      size_t channels, size_t kernel_width, size_t kernel_height,             \
      FixedBorderType border_type)

KLEIDICV_INSTANTIATE_TEMPLATE(int8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int32_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint32_t);
KLEIDICV_INSTANTIATE_TEMPLATE(float);

}  // namespace kleidicv::neon
