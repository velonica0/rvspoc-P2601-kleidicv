// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <utility>

#include "kleidicv/conversions/yuv_to_rgb.h"
#include "kleidicv/kleidicv.h"

#include "kleidicv/rvv.h"

#include "yuv444_coefficients.h"

namespace kleidicv::neon {

// Scalar YUV444 to RGB/BGR/RGBA/BGRA conversion using BT.601 coefficients.

static inline uint8_t clamp_u8(int32_t v) {
  return static_cast<uint8_t>(std::clamp(v, 0, 255));
}

template <bool BGR, bool kAlpha>
static kleidicv_error_t yuv444_to_rgbx_scalar(const uint8_t *src,
                                               size_t src_stride, uint8_t *dst,
                                               size_t dst_stride, size_t width,
                                               size_t height) {
  constexpr size_t dcn = kAlpha ? 4 : 3;
  constexpr size_t r_idx = BGR ? 2 : 0;
  constexpr size_t g_idx = 1;
  constexpr size_t b_idx = BGR ? 0 : 2;

  for (size_t h = 0; h < height; ++h) {
    const uint8_t *src_row = src + h * src_stride;
    uint8_t *dst_row = dst + h * dst_stride;

    for (size_t x = 0; x < width; ++x) {
      int32_t y = static_cast<int32_t>(src_row[x * 3 + 0]);
      int32_t u = static_cast<int32_t>(src_row[x * 3 + 1]);
      int32_t v = static_cast<int32_t>(src_row[x * 3 + 2]);

      int32_t b_val = y + rounding_shift_right((u - 128) * kUBWeight,
                                                kWeightScale);
      int32_t g_val =
          y + rounding_shift_right(
                  (u - 128) * kUGWeight + (v - 128) * kVGWeight, kWeightScale);
      int32_t r_val = y + rounding_shift_right((v - 128) * kVRWeight,
                                                kWeightScale);

      dst_row[x * dcn + r_idx] = clamp_u8(r_val);
      dst_row[x * dcn + g_idx] = clamp_u8(g_val);
      dst_row[x * dcn + b_idx] = clamp_u8(b_val);
      if constexpr (kAlpha) {
        dst_row[x * dcn + 3] = 0xFF;
      }
    }
  }
  return KLEIDICV_OK;
}

#ifdef __riscv_vector

// RVV vectorized YUV444 to RGB/BGR/RGBA/BGRA conversion using BT.601.
//
// Arithmetic is performed in i32 to accommodate kUBWeight=33292 which does not
// fit in a signed 16-bit value.  The pipeline is:
//   1. Segment-load 3 interleaved u8 channels (Y, U, V) with vlseg3e8.
//   2. Zero-extend u8m1 -> u16m2 -> u32m4, reinterpret as i32m4.
//   3. Compute colour offsets:  u_off = (int32)U - 128,  v_off = (int32)V - 128
//   4. Multiply-accumulate with BT.601 weights in i32, add rounding bias,
//      arithmetic-shift right by kWeightScale (14), add Y.
//   5. Clamp to [0, 255] and narrow i32m4 -> u16m2 -> u8m1.
//   6. Segment-store 3 or 4 channels (RGB / RGBA) with vsseg3/4e8.
//
// VLEN-agnostic: vl is obtained from vsetvl_e8m1 each iteration.

template <bool BGR, bool kAlpha>
static kleidicv_error_t yuv444_to_rgbx_rvv(const uint8_t *src,
                                            size_t src_stride, uint8_t *dst,
                                            size_t dst_stride, size_t width,
                                            size_t height) {
  constexpr size_t dcn = kAlpha ? 4 : 3;
  constexpr int32_t kRoundBias = 1 << (kWeightScale - 1);  // 1 << 13

  for (size_t h = 0; h < height; ++h) {
    const uint8_t *src_row = src + h * src_stride;
    uint8_t *dst_row = dst + h * dst_stride;
    size_t x = 0;

    while (x < width) {
      size_t vl = __riscv_vsetvl_e8m1(width - x);

      // 1. Segment load Y, U, V (interleaved u8x3)
      vuint8m1x3_t yuv_tuple = __riscv_vlseg3e8_v_u8m1x3(src_row + x * 3, vl);
      vuint8m1_t vy_u8 = __riscv_vget_v_u8m1x3_u8m1(yuv_tuple, 0);
      vuint8m1_t vu_u8 = __riscv_vget_v_u8m1x3_u8m1(yuv_tuple, 1);
      vuint8m1_t vv_u8 = __riscv_vget_v_u8m1x3_u8m1(yuv_tuple, 2);

      // 2. Widen to i32: u8m1 -> u16m2 -> u32m4 -> i32m4
      vuint16m2_t vy_u16 = __riscv_vzext_vf2_u16m2(vy_u8, vl);
      vuint16m2_t vu_u16 = __riscv_vzext_vf2_u16m2(vu_u8, vl);
      vuint16m2_t vv_u16 = __riscv_vzext_vf2_u16m2(vv_u8, vl);

      vuint32m4_t vy_u32 = __riscv_vzext_vf2_u32m4(vy_u16, vl);
      vuint32m4_t vu_u32 = __riscv_vzext_vf2_u32m4(vu_u16, vl);
      vuint32m4_t vv_u32 = __riscv_vzext_vf2_u32m4(vv_u16, vl);

      vint32m4_t y32 = __riscv_vreinterpret_v_u32m4_i32m4(vy_u32);
      vint32m4_t u32 = __riscv_vreinterpret_v_u32m4_i32m4(vu_u32);
      vint32m4_t v32 = __riscv_vreinterpret_v_u32m4_i32m4(vv_u32);

      // 3. u_off = U - 128, v_off = V - 128
      vint32m4_t u_off = __riscv_vsub_vx_i32m4(u32, 128, vl);
      vint32m4_t v_off = __riscv_vsub_vx_i32m4(v32, 128, vl);

      // 4. Compute B, G, R in i32 with rounding
      // b32 = Y + ((u_off * kUBWeight + kRoundBias) >> kWeightScale)
      vint32m4_t b_acc = __riscv_vmul_vx_i32m4(u_off, kUBWeight, vl);
      b_acc = __riscv_vadd_vx_i32m4(b_acc, kRoundBias, vl);
      b_acc = __riscv_vsra_vx_i32m4(b_acc, kWeightScale, vl);
      vint32m4_t b32 = __riscv_vadd_vv_i32m4(y32, b_acc, vl);

      // g32 = Y + ((u_off * kUGWeight + v_off * kVGWeight + kRoundBias) >> kWeightScale)
      vint32m4_t g_acc = __riscv_vmul_vx_i32m4(u_off, static_cast<int32_t>(kUGWeight), vl);
      g_acc = __riscv_vmacc_vx_i32m4(g_acc, static_cast<int32_t>(kVGWeight), v_off, vl);
      g_acc = __riscv_vadd_vx_i32m4(g_acc, kRoundBias, vl);
      g_acc = __riscv_vsra_vx_i32m4(g_acc, kWeightScale, vl);
      vint32m4_t g32 = __riscv_vadd_vv_i32m4(y32, g_acc, vl);

      // r32 = Y + ((v_off * kVRWeight + kRoundBias) >> kWeightScale)
      vint32m4_t r_acc = __riscv_vmul_vx_i32m4(v_off, static_cast<int32_t>(kVRWeight), vl);
      r_acc = __riscv_vadd_vx_i32m4(r_acc, kRoundBias, vl);
      r_acc = __riscv_vsra_vx_i32m4(r_acc, kWeightScale, vl);
      vint32m4_t r32 = __riscv_vadd_vv_i32m4(y32, r_acc, vl);

      // 5. Clamp to [0, 255] and narrow to u8
      r32 = __riscv_vmax_vx_i32m4(r32, 0, vl);
      r32 = __riscv_vmin_vx_i32m4(r32, 255, vl);
      g32 = __riscv_vmax_vx_i32m4(g32, 0, vl);
      g32 = __riscv_vmin_vx_i32m4(g32, 255, vl);
      b32 = __riscv_vmax_vx_i32m4(b32, 0, vl);
      b32 = __riscv_vmin_vx_i32m4(b32, 255, vl);

      // Narrow i32m4 -> u16m2 -> u8m1
      vuint32m4_t r_u32 = __riscv_vreinterpret_v_i32m4_u32m4(r32);
      vuint32m4_t g_u32 = __riscv_vreinterpret_v_i32m4_u32m4(g32);
      vuint32m4_t b_u32 = __riscv_vreinterpret_v_i32m4_u32m4(b32);

      vuint16m2_t r_u16 = __riscv_vnsrl_wx_u16m2(r_u32, 0, vl);
      vuint16m2_t g_u16 = __riscv_vnsrl_wx_u16m2(g_u32, 0, vl);
      vuint16m2_t b_u16 = __riscv_vnsrl_wx_u16m2(b_u32, 0, vl);

      vuint8m1_t r_u8 = __riscv_vnsrl_wx_u8m1(r_u16, 0, vl);
      vuint8m1_t g_u8 = __riscv_vnsrl_wx_u8m1(g_u16, 0, vl);
      vuint8m1_t b_u8 = __riscv_vnsrl_wx_u8m1(b_u16, 0, vl);

      // 6. Segment store with channel order depending on BGR/RGB and alpha
      if constexpr (BGR && kAlpha) {
        vuint8m1_t alpha = __riscv_vmv_v_x_u8m1(0xFF, vl);
        vuint8m1x4_t bgra = __riscv_vcreate_v_u8m1x4(b_u8, g_u8, r_u8, alpha);
        __riscv_vsseg4e8_v_u8m1x4(dst_row + x * dcn, bgra, vl);
      } else if constexpr (BGR && !kAlpha) {
        vuint8m1x3_t bgr = __riscv_vcreate_v_u8m1x3(b_u8, g_u8, r_u8);
        __riscv_vsseg3e8_v_u8m1x3(dst_row + x * dcn, bgr, vl);
      } else if constexpr (!BGR && kAlpha) {
        vuint8m1_t alpha = __riscv_vmv_v_x_u8m1(0xFF, vl);
        vuint8m1x4_t rgba = __riscv_vcreate_v_u8m1x4(r_u8, g_u8, b_u8, alpha);
        __riscv_vsseg4e8_v_u8m1x4(dst_row + x * dcn, rgba, vl);
      } else {
        vuint8m1x3_t rgb = __riscv_vcreate_v_u8m1x3(r_u8, g_u8, b_u8);
        __riscv_vsseg3e8_v_u8m1x3(dst_row + x * dcn, rgb, vl);
      }

      x += vl;
    }
  }
  return KLEIDICV_OK;
}

#endif  // __riscv_vector

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t yuv444_to_rgb_u8(const uint8_t *src, size_t src_stride,
                                  uint8_t *dst, size_t dst_stride, size_t width,
                                  size_t height,
                                  kleidicv_color_conversion_t color_format) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  switch (color_format) {
    case KLEIDICV_YUV444_TO_RGB:
#ifdef __riscv_vector
      return yuv444_to_rgbx_rvv<false, false>(src, src_stride, dst,
                                               dst_stride, width, height);
#else
      return yuv444_to_rgbx_scalar<false, false>(src, src_stride, dst,
                                                  dst_stride, width, height);
#endif
    case KLEIDICV_YUV444_TO_BGR:
#ifdef __riscv_vector
      return yuv444_to_rgbx_rvv<true, false>(src, src_stride, dst,
                                              dst_stride, width, height);
#else
      return yuv444_to_rgbx_scalar<true, false>(src, src_stride, dst,
                                                 dst_stride, width, height);
#endif
    case KLEIDICV_YUV444_TO_RGBA:
#ifdef __riscv_vector
      return yuv444_to_rgbx_rvv<false, true>(src, src_stride, dst,
                                              dst_stride, width, height);
#else
      return yuv444_to_rgbx_scalar<false, true>(src, src_stride, dst,
                                                 dst_stride, width, height);
#endif
    case KLEIDICV_YUV444_TO_BGRA:
#ifdef __riscv_vector
      return yuv444_to_rgbx_rvv<true, true>(src, src_stride, dst,
                                             dst_stride, width, height);
#else
      return yuv444_to_rgbx_scalar<true, true>(src, src_stride, dst,
                                                dst_stride, width, height);
#endif
    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
}

}  // namespace kleidicv::neon
