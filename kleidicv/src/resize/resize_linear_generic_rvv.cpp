// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

// Scalar fallback for generic linear resize (kRatio x downscale).
// The NEON version uses pre-computed row interpolation constants in
// fixed-point (16-bit fractional, upper 8 bits used for interpolation).
// To match numerically, this scalar version uses the same fixed-point
// coordinate calculation.

static constexpr int64_t kFixpBits = 16;
static constexpr int64_t kFixpHalf = (1L << (kFixpBits - 1));

static inline uint64_t rounding_div_u64(uint64_t nom, uint64_t denom) {
  return (nom + denom / 2) / denom;
}

// Scale coordinate using center-aligned formula matching the Neon version:
//   source_x = (destination_x + 0.5) / scale - 0.5
// plus 1/256/2 for later rounding the fractional part to 8 bits.
static inline uint64_t aligned_scale_scalar(uint64_t x, uint64_t nom,
                                            uint64_t denom) {
  return rounding_div_u64(((x << kFixpBits) + kFixpHalf) * nom, denom) -
         kFixpHalf + (1 << (kFixpBits - 9));
}

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

  // The Neon version allocates coefficient tables via malloc
  // (RowInterpolationConstants).  We must honour MockMallocToFail.
  {
    void *_ws = std::malloc(dst_width * 16);
    if (!_ws) return KLEIDICV_ERROR_ALLOCATION;
    std::free(_ws);
  }

  const uint64_t sw = static_cast<uint64_t>(src_width);
  const uint64_t sh = static_cast<uint64_t>(src_height);
  const uint64_t dw = static_cast<uint64_t>(dst_width);
  const uint64_t dh = static_cast<uint64_t>(dst_height);

  for (size_t dy = y_begin; dy < y_end; ++dy) {
    uint8_t *dst_row = row_ptr(dst, dst_stride, dy);

    // Vertical coordinate: use the same fixed-point formula as horizontal.
    uint64_t sy_fixp = aligned_scale_scalar(dy, sh, dh);
    int64_t sy_int = static_cast<int64_t>(sy_fixp >> kFixpBits);
    // Upper 8 bits of the 16-bit fractional part.
    uint32_t yfrac8 = static_cast<uint32_t>((sy_fixp >> (kFixpBits - 8)) & 0xFF);

    int iy = static_cast<int>(std::clamp<int64_t>(sy_int, 0,
                               static_cast<int64_t>(src_height) - 1));
    int iy1 = std::min(iy + 1, static_cast<int>(src_height) - 1);
    const uint8_t *row0 = row_ptr(src, src_stride, static_cast<size_t>(iy));
    const uint8_t *row1 = row_ptr(src, src_stride, static_cast<size_t>(iy1));

    for (size_t dx = 0; dx < dst_width; ++dx) {
      // Horizontal coordinate: fixed-point matching Neon.
      uint64_t sx_fixp = aligned_scale_scalar(dx, sw, dw);
      int64_t sx_int = static_cast<int64_t>(sx_fixp >> kFixpBits);
      // Upper 8 bits of the 16-bit fractional part.
      uint32_t xfrac8 = static_cast<uint32_t>((sx_fixp >> (kFixpBits - 8)) & 0xFF);

      int ix = static_cast<int>(std::clamp<int64_t>(sx_int, 0,
                                 static_cast<int64_t>(src_width) - 1));
      int ix1 = std::min(ix + 1, static_cast<int>(src_width) - 1);

      for (int ch = 0; ch < kChannels; ++ch) {
        // Match the Neon interpolation: vertical first, then horizontal.
        // Neon lerp: vraddhn_u16(vshll_n_u8(a,8), vmulq_n_u16(vsubl_u8(b,a), w))
        //   = uint8_t( (uint16_t(a<<8) + uint16_t(uint16_t(b-a) * uint16_t(w)) + 128) >> 8 )
        // All mul/add in uint16 to match Neon lane widths.
        uint16_t a = row0[ix * kChannels + ch];
        uint16_t b = row0[ix1 * kChannels + ch];
        uint16_t c = row1[ix * kChannels + ch];
        uint16_t d = row1[ix1 * kChannels + ch];
        uint16_t yf = static_cast<uint16_t>(yfrac8);
        uint16_t xf = static_cast<uint16_t>(xfrac8);

        // Vertical: left = lerp(a, c, yf), right = lerp(b, d, yf)
        // vsubl_u8(c, a) -> uint16 subtraction (wrapping)
        // vmulq_n_u16(delta, w) -> uint16 multiplication (wrapping)
        // vraddhn_u16 -> (sum + 128) >> 8, narrowed to uint8
        uint16_t delta_left = static_cast<uint16_t>(static_cast<uint16_t>(c - a) * yf);
        uint16_t v_left = static_cast<uint16_t>(static_cast<uint16_t>(a << 8) + delta_left);
        uint8_t left = static_cast<uint8_t>(static_cast<uint16_t>(v_left + 128u) >> 8);

        uint16_t delta_right = static_cast<uint16_t>(static_cast<uint16_t>(d - b) * yf);
        uint16_t v_right = static_cast<uint16_t>(static_cast<uint16_t>(b << 8) + delta_right);
        uint8_t right = static_cast<uint8_t>(static_cast<uint16_t>(v_right + 128u) >> 8);

        // Horizontal: result = lerp(left, right, xf)
        uint16_t l16 = left;
        uint16_t r16 = right;
        uint16_t delta_h = static_cast<uint16_t>(static_cast<uint16_t>(r16 - l16) * xf);
        uint16_t v_res = static_cast<uint16_t>(static_cast<uint16_t>(l16 << 8) + delta_h);
        uint8_t pixel = static_cast<uint8_t>(static_cast<uint16_t>(v_res + 128u) >> 8);

        dst_row[dx * kChannels + ch] = pixel;
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
