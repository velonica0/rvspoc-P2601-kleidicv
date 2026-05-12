// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <limits>
#include <utility>

#include "kleidicv/conversions/rgb_to_yuv.h"
#include "kleidicv/kleidicv.h"
#include "kleidicv/utils.h"

#include "kleidicv/rvv.h"

#include "rgb_to_yuv444_coefficients.h"

namespace kleidicv::neon {

// half_ = (max_uint8/2 + 1) << kWeightScale = 128 << 14
static constexpr uint32_t kHalf =
    (std::numeric_limits<uint8_t>::max() / 2 + 1U) << kWeightScale;

template <bool BGR, bool kAlpha>
static kleidicv_error_t rgb_to_yuv444_scalar(const uint8_t *src,
                                             size_t src_stride, uint8_t *dst,
                                             size_t dst_stride, size_t width,
                                             size_t height) {
  constexpr size_t r_idx = BGR ? 2 : 0;
  constexpr size_t g_idx = 1;
  constexpr size_t b_idx = BGR ? 0 : 2;
  constexpr size_t scn = kAlpha ? 4 : 3;
  constexpr size_t dcn = 3;

  for (size_t h = 0; h < height; ++h) {
    const uint8_t *src_row = src + src_stride * h;
    uint8_t *dst_row = dst + dst_stride * h;

    for (size_t x = 0; x < width; ++x) {
      int32_t r = static_cast<int32_t>(src_row[x * scn + r_idx]);
      int32_t g = static_cast<int32_t>(src_row[x * scn + g_idx]);
      int32_t b = static_cast<int32_t>(src_row[x * scn + b_idx]);

      // Compute Y: matches neon scalar_path exactly
      int32_t y = r * kRYWeight + g * kGYWeight + b * kBYWeight;
      y = rounding_shift_right(y, kWeightScale);

      // Compute U = (B - Y) * kBUWeight + half_
      int32_t u = (b - y) * kBUWeight + static_cast<int32_t>(kHalf);
      u = rounding_shift_right(u, kWeightScale);

      // Compute V = (R - Y) * kRVWeight + half_
      int32_t v = (r - y) * kRVWeight + static_cast<int32_t>(kHalf);
      v = rounding_shift_right(v, kWeightScale);

      dst_row[x * dcn + 0] = saturating_cast<int32_t, uint8_t>(y);
      dst_row[x * dcn + 1] = saturating_cast<int32_t, uint8_t>(u);
      dst_row[x * dcn + 2] = saturating_cast<int32_t, uint8_t>(v);
    }
  }

  return KLEIDICV_OK;
}

#ifdef __riscv_vector

// RVV vectorized RGB/BGR/RGBA/BGRA to YUV444 conversion using BT.601.
//
// All weights (kRYWeight, kGYWeight, kBYWeight, kBUWeight, kRVWeight) fit in
// signed 16-bit, but the intermediate products require i32.  Pipeline:
//   1. Segment-load 3 or 4 interleaved u8 channels with vlseg3/4e8.
//   2. Zero-extend u8m1 -> u16m2 -> u32m4, reinterpret as i32m4.
//   3. Compute Y  = rounding_shift_right(R*kRYWeight + G*kGYWeight + B*kBYWeight, 14)
//   4. Compute U  = rounding_shift_right((B - Y)*kBUWeight + kHalf, 14)
//   5. Compute V  = rounding_shift_right((R - Y)*kRVWeight + kHalf, 14)
//   6. Clamp to [0,255], narrow i32m4 -> u16m2 -> u8m1.
//   7. Segment-store 3 channels [Y, U, V] with vsseg3e8.
//
// VLEN-agnostic: vl is obtained from vsetvl_e8m1 each iteration.

template <bool BGR, bool kAlpha>
static kleidicv_error_t rgb_to_yuv444_rvv(const uint8_t *src,
                                           size_t src_stride, uint8_t *dst,
                                           size_t dst_stride, size_t width,
                                           size_t height) {
  constexpr size_t scn = kAlpha ? 4 : 3;
  constexpr int32_t kRoundBias = 1 << (kWeightScale - 1);  // 1 << 13
  constexpr int32_t kHalf32 = static_cast<int32_t>(kHalf);

  for (size_t h = 0; h < height; ++h) {
    const uint8_t *src_row = src + src_stride * h;
    uint8_t *dst_row = dst + dst_stride * h;
    size_t x = 0;

    while (x < width) {
      size_t vl = __riscv_vsetvl_e8m1(width - x);

      // 1. Segment load R, G, B (and optionally A)
      vuint8m1_t vr_u8, vg_u8, vb_u8;

      if constexpr (kAlpha) {
        vuint8m1x4_t rgba = __riscv_vlseg4e8_v_u8m1x4(src_row + x * scn, vl);
        if constexpr (BGR) {
          vb_u8 = __riscv_vget_v_u8m1x4_u8m1(rgba, 0);
          vg_u8 = __riscv_vget_v_u8m1x4_u8m1(rgba, 1);
          vr_u8 = __riscv_vget_v_u8m1x4_u8m1(rgba, 2);
          // alpha channel (index 3) is ignored
        } else {
          vr_u8 = __riscv_vget_v_u8m1x4_u8m1(rgba, 0);
          vg_u8 = __riscv_vget_v_u8m1x4_u8m1(rgba, 1);
          vb_u8 = __riscv_vget_v_u8m1x4_u8m1(rgba, 2);
        }
      } else {
        vuint8m1x3_t rgb = __riscv_vlseg3e8_v_u8m1x3(src_row + x * scn, vl);
        if constexpr (BGR) {
          vb_u8 = __riscv_vget_v_u8m1x3_u8m1(rgb, 0);
          vg_u8 = __riscv_vget_v_u8m1x3_u8m1(rgb, 1);
          vr_u8 = __riscv_vget_v_u8m1x3_u8m1(rgb, 2);
        } else {
          vr_u8 = __riscv_vget_v_u8m1x3_u8m1(rgb, 0);
          vg_u8 = __riscv_vget_v_u8m1x3_u8m1(rgb, 1);
          vb_u8 = __riscv_vget_v_u8m1x3_u8m1(rgb, 2);
        }
      }

      // 2. Widen to i32: u8m1 -> u16m2 -> u32m4 -> i32m4
      vint32m4_t r32 = __riscv_vreinterpret_v_u32m4_i32m4(
          __riscv_vzext_vf2_u32m4(__riscv_vzext_vf2_u16m2(vr_u8, vl), vl));
      vint32m4_t g32 = __riscv_vreinterpret_v_u32m4_i32m4(
          __riscv_vzext_vf2_u32m4(__riscv_vzext_vf2_u16m2(vg_u8, vl), vl));
      vint32m4_t b32 = __riscv_vreinterpret_v_u32m4_i32m4(
          __riscv_vzext_vf2_u32m4(__riscv_vzext_vf2_u16m2(vb_u8, vl), vl));

      // 3. Y = rounding_shift_right(R*kRYWeight + G*kGYWeight + B*kBYWeight, 14)
      vint32m4_t y_acc = __riscv_vmul_vx_i32m4(r32, static_cast<int32_t>(kRYWeight), vl);
      y_acc = __riscv_vmacc_vx_i32m4(y_acc, static_cast<int32_t>(kGYWeight), g32, vl);
      y_acc = __riscv_vmacc_vx_i32m4(y_acc, static_cast<int32_t>(kBYWeight), b32, vl);
      y_acc = __riscv_vadd_vx_i32m4(y_acc, kRoundBias, vl);
      vint32m4_t y32 = __riscv_vsra_vx_i32m4(y_acc, kWeightScale, vl);

      // 4. U = rounding_shift_right((B - Y) * kBUWeight + kHalf, 14)
      vint32m4_t b_minus_y = __riscv_vsub_vv_i32m4(b32, y32, vl);
      vint32m4_t u_acc = __riscv_vmul_vx_i32m4(b_minus_y, static_cast<int32_t>(kBUWeight), vl);
      u_acc = __riscv_vadd_vx_i32m4(u_acc, kHalf32, vl);
      u_acc = __riscv_vadd_vx_i32m4(u_acc, kRoundBias, vl);
      vint32m4_t u32 = __riscv_vsra_vx_i32m4(u_acc, kWeightScale, vl);

      // 5. V = rounding_shift_right((R - Y) * kRVWeight + kHalf, 14)
      vint32m4_t r_minus_y = __riscv_vsub_vv_i32m4(r32, y32, vl);
      vint32m4_t v_acc = __riscv_vmul_vx_i32m4(r_minus_y, static_cast<int32_t>(kRVWeight), vl);
      v_acc = __riscv_vadd_vx_i32m4(v_acc, kHalf32, vl);
      v_acc = __riscv_vadd_vx_i32m4(v_acc, kRoundBias, vl);
      vint32m4_t v32 = __riscv_vsra_vx_i32m4(v_acc, kWeightScale, vl);

      // 6. Clamp to [0, 255] and narrow to u8
      y32 = __riscv_vmax_vx_i32m4(y32, 0, vl);
      y32 = __riscv_vmin_vx_i32m4(y32, 255, vl);
      u32 = __riscv_vmax_vx_i32m4(u32, 0, vl);
      u32 = __riscv_vmin_vx_i32m4(u32, 255, vl);
      v32 = __riscv_vmax_vx_i32m4(v32, 0, vl);
      v32 = __riscv_vmin_vx_i32m4(v32, 255, vl);

      // Narrow i32m4 -> u16m2 -> u8m1
      vuint8m1_t y_u8 = __riscv_vnsrl_wx_u8m1(
          __riscv_vnsrl_wx_u16m2(__riscv_vreinterpret_v_i32m4_u32m4(y32), 0, vl),
          0, vl);
      vuint8m1_t u_u8 = __riscv_vnsrl_wx_u8m1(
          __riscv_vnsrl_wx_u16m2(__riscv_vreinterpret_v_i32m4_u32m4(u32), 0, vl),
          0, vl);
      vuint8m1_t v_u8 = __riscv_vnsrl_wx_u8m1(
          __riscv_vnsrl_wx_u16m2(__riscv_vreinterpret_v_i32m4_u32m4(v32), 0, vl),
          0, vl);

      // 7. Segment store [Y, U, V]
      vuint8m1x3_t yuv_out = __riscv_vcreate_v_u8m1x3(y_u8, u_u8, v_u8);
      __riscv_vsseg3e8_v_u8m1x3(dst_row + x * 3, yuv_out, vl);

      x += vl;
    }
  }

  return KLEIDICV_OK;
}

#endif  // __riscv_vector

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgb_to_yuv444_u8(const uint8_t *src, size_t src_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height,
                                  kleidicv_color_conversion_t color_format) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  switch (color_format) {
    case KLEIDICV_RGB_TO_YUV444:
#ifdef __riscv_vector
      return rgb_to_yuv444_rvv<false, false>(src, src_stride, dst,
                                              dst_stride, width, height);
#else
      return rgb_to_yuv444_scalar<false, false>(src, src_stride, dst,
                                                dst_stride, width, height);
#endif
    case KLEIDICV_BGR_TO_YUV444:
#ifdef __riscv_vector
      return rgb_to_yuv444_rvv<true, false>(src, src_stride, dst, dst_stride,
                                             width, height);
#else
      return rgb_to_yuv444_scalar<true, false>(src, src_stride, dst, dst_stride,
                                               width, height);
#endif
    case KLEIDICV_RGBA_TO_YUV444:
#ifdef __riscv_vector
      return rgb_to_yuv444_rvv<false, true>(src, src_stride, dst, dst_stride,
                                             width, height);
#else
      return rgb_to_yuv444_scalar<false, true>(src, src_stride, dst, dst_stride,
                                               width, height);
#endif
    case KLEIDICV_BGRA_TO_YUV444:
#ifdef __riscv_vector
      return rgb_to_yuv444_rvv<true, true>(src, src_stride, dst, dst_stride,
                                            width, height);
#else
      return rgb_to_yuv444_scalar<true, true>(src, src_stride, dst, dst_stride,
                                              width, height);
#endif
    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
}

}  // namespace kleidicv::neon
