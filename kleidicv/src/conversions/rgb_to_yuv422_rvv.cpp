// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstddef>
#include <utility>

#include "kleidicv/conversions/rgb_to_yuv.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/utils.h"

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

static const int kWeightScale422 = 14;

static const int KR2Y422Weight = 4211;
static const int KG2Y422Weight = 8258;
static const int KB2Y422Weight = 1606;

static const int KR2U422Weight = -1212;
static const int KG2U422Weight = -2384;
static const int KB2U422Weight = 3596;
static const int KG2V422Weight = -3015;
static const int KB2V422Weight = -582;

template <size_t b_idx, size_t u_idx, size_t y_idx, size_t scn>
static kleidicv_error_t rgb_to_yuv422_scalar(const uint8_t *src,
                                             size_t src_stride, uint8_t *dst,
                                             size_t dst_stride, size_t width,
                                             size_t height) {
  constexpr size_t r_idx = 2 - b_idx;
  constexpr size_t v_idx = (u_idx + 2) % 4;
  constexpr size_t dcn = 2;

  for (size_t h = 0; h < height; ++h) {
    const uint8_t *src_row = src + src_stride * h;
    uint8_t *dst_row = dst + dst_stride * h;

    for (size_t x = 0; x < width; x += 2) {
      uint8_t r1 = src_row[x * scn + r_idx];
      uint8_t g1 = src_row[x * scn + 1];
      uint8_t b1 = src_row[x * scn + b_idx];
      uint8_t r2 = src_row[(x + 1) * scn + r_idx];
      uint8_t g2 = src_row[(x + 1) * scn + 1];
      uint8_t b2 = src_row[(x + 1) * scn + b_idx];

      // Compute Y per pixel
      int y1_ = r1 * KR2Y422Weight + g1 * KG2Y422Weight + b1 * KB2Y422Weight +
                (1 << kWeightScale422) * 16;
      uint8_t y0_val = saturating_cast<int, uint8_t>(
          ((1 << (kWeightScale422 - 1)) + y1_) >> kWeightScale422);

      int y2_ = r2 * KR2Y422Weight + g2 * KG2Y422Weight + b2 * KB2Y422Weight +
                (1 << kWeightScale422) * 16;
      uint8_t y1_val = saturating_cast<int, uint8_t>(
          ((1 << (kWeightScale422 - 1)) + y2_) >> kWeightScale422);

      // Compute U and V from the sum of the two pixels (not average)
      int sr = r1 + r2, sg = g1 + g2, sb = b1 + b2;

      int u_ = sr * KR2U422Weight + sg * KG2U422Weight + sb * KB2U422Weight +
               (1 << (kWeightScale422 - 1)) * 256;
      uint8_t u_val = saturating_cast<int, uint8_t>(
          ((1 << (kWeightScale422 - 1)) + u_) >> kWeightScale422);

      int v_ = sr * KB2U422Weight + sg * KG2V422Weight + sb * KB2V422Weight +
               (1 << (kWeightScale422 - 1)) * 256;
      uint8_t v_val = saturating_cast<int, uint8_t>(
          ((1 << (kWeightScale422 - 1)) + v_) >> kWeightScale422);

      // Pack into YUV422 format: 4 bytes per pixel pair
      // The output layout depends on the template parameters:
      //   YUYV: y_idx=0, u_idx=1 => [Y0, U, Y1, V]
      //   UYVY: y_idx=1, u_idx=0 => [U, Y0, V, Y1]
      //   YVYU: y_idx=0, u_idx=3 => [Y0, V, Y1, U]
      dst_row[x * dcn + y_idx] = y0_val;
      dst_row[x * dcn + y_idx + 2] = y1_val;
      dst_row[x * dcn + u_idx] = u_val;
      dst_row[x * dcn + v_idx] = v_val;
    }
  }

  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgb_to_yuv422_u8(const uint8_t *src, size_t src_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height,
                                  kleidicv_color_conversion_t color_format) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  if (width < 2 || (width % 2) != 0) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  switch (color_format) {
    case KLEIDICV_BGR_TO_YUYV:
      return rgb_to_yuv422_scalar<0, 1, 0, 3>(src, src_stride, dst, dst_stride,
                                               width, height);
    case KLEIDICV_BGR_TO_UYVY:
      return rgb_to_yuv422_scalar<0, 0, 1, 3>(src, src_stride, dst, dst_stride,
                                               width, height);
    case KLEIDICV_BGR_TO_YVYU:
      return rgb_to_yuv422_scalar<0, 3, 0, 3>(src, src_stride, dst, dst_stride,
                                               width, height);
    case KLEIDICV_RGB_TO_YUYV:
      return rgb_to_yuv422_scalar<2, 1, 0, 3>(src, src_stride, dst, dst_stride,
                                               width, height);
    case KLEIDICV_RGB_TO_UYVY:
      return rgb_to_yuv422_scalar<2, 0, 1, 3>(src, src_stride, dst, dst_stride,
                                               width, height);
    case KLEIDICV_RGB_TO_YVYU:
      return rgb_to_yuv422_scalar<2, 3, 0, 3>(src, src_stride, dst, dst_stride,
                                               width, height);
    case KLEIDICV_BGRA_TO_YUYV:
      return rgb_to_yuv422_scalar<0, 1, 0, 4>(src, src_stride, dst, dst_stride,
                                               width, height);
    case KLEIDICV_BGRA_TO_UYVY:
      return rgb_to_yuv422_scalar<0, 0, 1, 4>(src, src_stride, dst, dst_stride,
                                               width, height);
    case KLEIDICV_BGRA_TO_YVYU:
      return rgb_to_yuv422_scalar<0, 3, 0, 4>(src, src_stride, dst, dst_stride,
                                               width, height);
    case KLEIDICV_RGBA_TO_YUYV:
      return rgb_to_yuv422_scalar<2, 1, 0, 4>(src, src_stride, dst, dst_stride,
                                               width, height);
    case KLEIDICV_RGBA_TO_UYVY:
      return rgb_to_yuv422_scalar<2, 0, 1, 4>(src, src_stride, dst, dst_stride,
                                               width, height);
    case KLEIDICV_RGBA_TO_YVYU:
      return rgb_to_yuv422_scalar<2, 3, 0, 4>(src, src_stride, dst, dst_stride,
                                               width, height);
    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
}

}  // namespace kleidicv::neon
