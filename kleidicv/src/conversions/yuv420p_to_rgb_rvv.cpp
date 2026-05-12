// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <utility>

#include "kleidicv/conversions/yuv_to_rgb.h"
#include "kleidicv/kleidicv.h"

#include "kleidicv/rvv.h"

#include "yuv42x_coefficients.h"

namespace kleidicv::neon {

// Scalar YUV420p to RGB/BGR/RGBA/BGRA conversion.
// Uses the same BT.601 coefficients as the neon implementation.

static inline uint8_t clamp_u8(int32_t v) {
  return static_cast<uint8_t>(std::clamp(v, 0, 255));
}

template <bool BGR, bool kAlpha>
static void yuv420p_to_rgbx_scalar(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, bool v_first, size_t begin, size_t end) {
  constexpr size_t dcn = kAlpha ? 4 : 3;

  const uint8_t *y_plane = src;
  const uint8_t *u_plane = src + src_stride * height;
  const uint8_t *v_plane =
      u_plane + src_stride * (height / 4) + (width / 2) * ((height % 4) / 2);

  if (v_first) {
    std::swap(u_plane, v_plane);
  }

  size_t u_index = 0;
  size_t v_index = height % 4 == 2 ? 1 : 0;

  size_t row_begin = begin * 2;
  size_t row_end = std::min<size_t>(height, end * 2);
  size_t row_uv = begin;

  size_t uv_strides[2] = {width / 2, src_stride - width / 2};

  const uint8_t *y0 = y_plane + row_begin * src_stride;
  const uint8_t *u = u_plane + (row_uv / 2) * src_stride;
  const uint8_t *v = v_plane + (row_uv / 2) * src_stride;

  if (row_uv % 2 == 1) {
    u += uv_strides[(u_index++) & 1];
    v += uv_strides[(v_index++) & 1];
  }

  for (size_t h = row_begin; h < row_end; h += 2) {
    uint8_t *row0 = dst + dst_stride * h;
    uint8_t *row1 = dst + dst_stride * (h + 1);
    const uint8_t *y1 = y0 + src_stride;

    if (h == (row_end - 1)) {
      row1 = row0;
      y1 = y0;
    }

    for (size_t x = 0; x < width; ++x) {
      int32_t u_m128 = static_cast<int32_t>(u[x / 2]) - 128;
      int32_t v_m128 = static_cast<int32_t>(v[x / 2]) - 128;

      for (int sel = 0; sel < 2; ++sel) {
        const uint8_t *yp = (sel == 0) ? y0 : y1;
        uint8_t *outp = (sel == 0) ? row0 : row1;

        int32_t yv = kYWeight * std::max(static_cast<int>(yp[x]) - 16, 0);
        int32_t r = yv + kUVWeights[kRVWeightIndex] * v_m128;
        int32_t g = yv + kUVWeights[kGUWeightIndex] * u_m128 +
                    kUVWeights[kGVWeightIndex] * v_m128;
        int32_t b = yv + kUVWeights[kBUWeightIndex] * u_m128;

        r = (r + (1 << (kWeightScale - 1))) >> kWeightScale;
        g = (g + (1 << (kWeightScale - 1))) >> kWeightScale;
        b = (b + (1 << (kWeightScale - 1))) >> kWeightScale;

        if constexpr (BGR) {
          outp[x * dcn + 0] = clamp_u8(b);
          outp[x * dcn + 1] = clamp_u8(g);
          outp[x * dcn + 2] = clamp_u8(r);
        } else {
          outp[x * dcn + 0] = clamp_u8(r);
          outp[x * dcn + 1] = clamp_u8(g);
          outp[x * dcn + 2] = clamp_u8(b);
        }
        if constexpr (kAlpha) {
          outp[x * dcn + 3] = 0xFF;
        }
      }
    }

    y0 += src_stride * 2;
    u += uv_strides[(u_index++) & 1];
    v += uv_strides[(v_index++) & 1];
  }
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t yuv420p_to_rgb_stripe_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, kleidicv_color_conversion_t color_format,
    size_t begin, size_t end) {
  // Check color_format validity BEFORE image-size checks, matching the Neon
  // implementation order.  This ensures invalid formats are rejected with
  // NOT_IMPLEMENTED even when the image dimensions would also fail range checks.
  switch (color_format) {
    case KLEIDICV_YV12_TO_BGR:
    case KLEIDICV_YV12_TO_RGB:
    case KLEIDICV_YV12_TO_BGRA:
    case KLEIDICV_YV12_TO_RGBA:
    case KLEIDICV_IYUV_TO_BGR:
    case KLEIDICV_IYUV_TO_RGB:
    case KLEIDICV_IYUV_TO_BGRA:
    case KLEIDICV_IYUV_TO_RGBA:
      break;
    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  CHECK_POINTER_AND_STRIDE(src, src_stride, (height * 3 + 1) / 2);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  switch (color_format) {
    case KLEIDICV_YV12_TO_BGR:
      yuv420p_to_rgbx_scalar<true, false>(src, src_stride, dst, dst_stride,
                                          width, height, true, begin, end);
      return KLEIDICV_OK;
    case KLEIDICV_YV12_TO_RGB:
      yuv420p_to_rgbx_scalar<false, false>(src, src_stride, dst, dst_stride,
                                           width, height, true, begin, end);
      return KLEIDICV_OK;
    case KLEIDICV_YV12_TO_BGRA:
      yuv420p_to_rgbx_scalar<true, true>(src, src_stride, dst, dst_stride,
                                         width, height, true, begin, end);
      return KLEIDICV_OK;
    case KLEIDICV_YV12_TO_RGBA:
      yuv420p_to_rgbx_scalar<false, true>(src, src_stride, dst, dst_stride,
                                          width, height, true, begin, end);
      return KLEIDICV_OK;
    case KLEIDICV_IYUV_TO_BGR:
      yuv420p_to_rgbx_scalar<true, false>(src, src_stride, dst, dst_stride,
                                          width, height, false, begin, end);
      return KLEIDICV_OK;
    case KLEIDICV_IYUV_TO_RGB:
      yuv420p_to_rgbx_scalar<false, false>(src, src_stride, dst, dst_stride,
                                           width, height, false, begin, end);
      return KLEIDICV_OK;
    case KLEIDICV_IYUV_TO_BGRA:
      yuv420p_to_rgbx_scalar<true, true>(src, src_stride, dst, dst_stride,
                                         width, height, false, begin, end);
      return KLEIDICV_OK;
    case KLEIDICV_IYUV_TO_RGBA:
      yuv420p_to_rgbx_scalar<false, true>(src, src_stride, dst, dst_stride,
                                          width, height, false, begin, end);
      return KLEIDICV_OK;
    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
}

}  // namespace kleidicv::neon
