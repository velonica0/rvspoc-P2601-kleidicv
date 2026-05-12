// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

// Scalar fallback for generic linear resize (kRatio x upscale).
// The NEON version uses pre-computed row interpolation constants and
// vectorized horizontal interpolation.

template <int kRatio, int kChannels>
kleidicv_error_t kleidicv_resize_generic_stripe_u8(
    const uint8_t *src, size_t src_stride, size_t src_width, size_t src_height,
    size_t y_begin, size_t y_end, uint8_t *dst, size_t dst_stride,
    size_t dst_width, size_t dst_height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, dst_height);

  if (src_width == 0 || src_height == 0) {
    return KLEIDICV_OK;
  }

  for (size_t dy = y_begin; dy < y_end; ++dy) {
    uint8_t *dst_row = row_ptr(dst, dst_stride, dy);
    float fy =
        (static_cast<float>(dy) + 0.5f) *
            static_cast<float>(src_height) / static_cast<float>(dst_height) -
        0.5f;
    int iy = static_cast<int>(std::floor(fy));
    float yfrac = fy - static_cast<float>(iy);
    iy = std::clamp(iy, 0, static_cast<int>(src_height) - 1);
    int iy1 = std::min(iy + 1, static_cast<int>(src_height) - 1);
    const uint8_t *row0 = row_ptr(src, src_stride, static_cast<size_t>(iy));
    const uint8_t *row1 = row_ptr(src, src_stride, static_cast<size_t>(iy1));

    for (size_t dx = 0; dx < dst_width; ++dx) {
      for (int ch = 0; ch < kChannels; ++ch) {
        float fx =
            (static_cast<float>(dx) + 0.5f) *
                static_cast<float>(src_width) /
                static_cast<float>(dst_width) -
            0.5f;
        int ix = static_cast<int>(std::floor(fx));
        float xfrac = fx - static_cast<float>(ix);
        ix = std::clamp(ix, 0, static_cast<int>(src_width) - 1);
        int ix1 = std::min(ix + 1, static_cast<int>(src_width) - 1);

        float a = static_cast<float>(row0[ix * kChannels + ch]);
        float b = static_cast<float>(row0[ix1 * kChannels + ch]);
        float c = static_cast<float>(row1[ix * kChannels + ch]);
        float d = static_cast<float>(row1[ix1 * kChannels + ch]);
        float val = a * (1 - xfrac) * (1 - yfrac) +
                    b * xfrac * (1 - yfrac) + c * (1 - xfrac) * yfrac +
                    d * xfrac * yfrac;
        dst_row[dx * kChannels + ch] =
            static_cast<uint8_t>(std::clamp(val + 0.5f, 0.0f, 255.0f));
      }
    }
  }
  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(ratio, channels)                      \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t                        \
  kleidicv_resize_generic_stripe_u8<ratio, channels>(                       \
      const uint8_t *src, size_t src_stride, size_t src_width,              \
      size_t src_height, size_t y_begin, size_t y_end, uint8_t *dst,        \
      size_t dst_stride, size_t dst_width, size_t dst_height)

KLEIDICV_INSTANTIATE_TEMPLATE(2, 1);
KLEIDICV_INSTANTIATE_TEMPLATE(2, 2);
KLEIDICV_INSTANTIATE_TEMPLATE(2, 3);
KLEIDICV_INSTANTIATE_TEMPLATE(3, 1);
KLEIDICV_INSTANTIATE_TEMPLATE(3, 2);
KLEIDICV_INSTANTIATE_TEMPLATE(3, 3);

}  // namespace kleidicv::neon
