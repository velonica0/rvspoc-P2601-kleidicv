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

// Scalar YUV422 to RGB/BGR/RGBA/BGRA conversion using BT.601 coefficients.

static inline uint8_t clamp_u8(int32_t v) {
  return static_cast<uint8_t>(std::clamp(v, 0, 255));
}

template <size_t b_idx, size_t u_chroma_idx, size_t y_idx, size_t dcn>
static kleidicv_error_t yuv422_to_rgbx_scalar(const uint8_t *src,
                                               size_t src_stride, uint8_t *dst,
                                               size_t dst_stride, size_t width,
                                               size_t height) {
  constexpr size_t u_idx = u_chroma_idx;
  constexpr size_t v_idx = (u_idx + 2) % 4;
  constexpr size_t scn = 2;  // source channels per pixel on average

  for (size_t h = 0; h < height; ++h) {
    const uint8_t *src_row = src + h * src_stride;
    uint8_t *dst_row = dst + h * dst_stride;

    for (size_t x = 0; x < width; x += 2) {
      uint8_t u_val = src_row[x * scn + u_idx];
      uint8_t v_val = src_row[x * scn + v_idx];
      uint8_t y0_val = src_row[x * scn + y_idx];
      uint8_t y1_val = src_row[x * scn + y_idx + scn];

      int32_t u_m128 = static_cast<int32_t>(u_val) - 128;
      int32_t v_m128 = static_cast<int32_t>(v_val) - 128;

      int32_t r_sub_y = kUVWeights[kRVWeightIndex] * v_m128 +
                         (1 << (kWeightScale - 1));
      int32_t g_sub_y = kUVWeights[kGUWeightIndex] * u_m128 +
                         kUVWeights[kGVWeightIndex] * v_m128 +
                         (1 << (kWeightScale - 1));
      int32_t b_sub_y = kUVWeights[kBUWeightIndex] * u_m128 +
                         (1 << (kWeightScale - 1));

      uint8_t y_vals[2] = {y0_val, y1_val};
      uint8_t *out_ptrs[2] = {dst_row + x * dcn, dst_row + (x + 1) * dcn};

      for (size_t sel = 0; sel < 2; ++sel) {
        int32_t y = kYWeight * std::max(static_cast<int>(y_vals[sel]) - 16, 0);
        int32_t r = (y + r_sub_y) >> kWeightScale;
        int32_t g = (y + g_sub_y) >> kWeightScale;
        int32_t b = (y + b_sub_y) >> kWeightScale;

        out_ptrs[sel][2 - b_idx] = clamp_u8(r);
        out_ptrs[sel][1] = clamp_u8(g);
        out_ptrs[sel][b_idx] = clamp_u8(b);

        if constexpr (dcn > 3) {
          out_ptrs[sel][3] = 0xFF;
        }
      }
    }
  }
  return KLEIDICV_OK;
}

#ifdef __riscv_vector

// RVV vectorized YUV422 to RGB/BGR/RGBA/BGRA conversion using BT.601.
//
// YUV422 packs 2 pixels into every 4 bytes.  The byte order within each
// 4-byte group depends on the format:
//   YUYV:  [Y0, U, Y1, V]  -> y_idx=0, u_chroma_idx=1
//   UYVY:  [U, Y0, V, Y1]  -> y_idx=1, u_chroma_idx=0
//   YVYU:  [Y0, V, Y1, U]  -> y_idx=0, u_chroma_idx=3
//
// Strategy:
//   1. Use vlseg4e8 to deinterleave N groups of 4 bytes into 4 channels.
//   2. Map channels to Y_even, Y_odd, U, V based on template parameters.
//   3. Compute colour offsets from U, V (shared between even/odd pixels).
//   4. Process Y_even and Y_odd separately against the shared UV offsets.
//   5. Use strided segment stores (vssseg3e8 / vssseg4e8) with stride = 2*dcn
//      to write even pixels to positions 0, 2, 4, ... and odd pixels to
//      positions 1, 3, 5, ... within each row.
//
// Template parameters:
//   b_idx: 0 for BGR output, 2 for RGB output
//   u_chroma_idx: position of U in the 4-byte group (0, 1, or 3)
//   y_idx: position of first Y in the 4-byte group (0 or 1)
//   dcn: output channels per pixel (3 or 4)
//
// VLEN-agnostic: vl is obtained from vsetvl each iteration.

// Helper: compute R, G, B u8 vectors from Y u8 values and precomputed
// UV colour offsets, then perform a strided segment store.
template <size_t b_idx, size_t dcn>
static inline void yuv422_row_store_rvv(
    const vuint8m1_t &vy_u8, uint8_t *outp, ptrdiff_t stride, size_t vl,
    const vint32m4_t &r_sub_y, const vint32m4_t &g_sub_y,
    const vint32m4_t &b_sub_y) {
  // Y' = max(Y - 16, 0)
  vuint8m1_t vy_sub16 = __riscv_vssubu_vx_u8m1(vy_u8, 16, vl);

  // Widen u8m1 -> u16m2 -> u32m4 -> i32m4
  vuint16m2_t vy_u16 = __riscv_vzext_vf2_u16m2(vy_sub16, vl);
  vuint32m4_t vy_u32 = __riscv_vzext_vf2_u32m4(vy_u16, vl);
  vint32m4_t y32 = __riscv_vreinterpret_v_u32m4_i32m4(vy_u32);

  // y_weighted = kYWeight * Y'
  vint32m4_t y_weighted = __riscv_vmul_vx_i32m4(y32, kYWeight, vl);

  // R = (y_weighted + r_sub_y) >> kWeightScale
  vint32m4_t r32 = __riscv_vadd_vv_i32m4(y_weighted, r_sub_y, vl);
  r32 = __riscv_vsra_vx_i32m4(r32, kWeightScale, vl);

  // G = (y_weighted + g_sub_y) >> kWeightScale
  vint32m4_t g32 = __riscv_vadd_vv_i32m4(y_weighted, g_sub_y, vl);
  g32 = __riscv_vsra_vx_i32m4(g32, kWeightScale, vl);

  // B = (y_weighted + b_sub_y) >> kWeightScale
  vint32m4_t b32 = __riscv_vadd_vv_i32m4(y_weighted, b_sub_y, vl);
  b32 = __riscv_vsra_vx_i32m4(b32, kWeightScale, vl);

  // Clamp to [0, 255]
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

  // Strided segment store: write vl pixels at positions
  // outp[0], outp[stride], outp[2*stride], ...
  // Each pixel is dcn bytes written as an interleaved segment.
  if constexpr (b_idx == 0 && dcn == 4) {
    // BGR + Alpha
    vuint8m1_t alpha = __riscv_vmv_v_x_u8m1(0xFF, vl);
    vuint8m1x4_t bgra = __riscv_vcreate_v_u8m1x4(b_u8, g_u8, r_u8, alpha);
    __riscv_vssseg4e8_v_u8m1x4(outp, stride, bgra, vl);
  } else if constexpr (b_idx == 0 && dcn == 3) {
    // BGR
    vuint8m1x3_t bgr = __riscv_vcreate_v_u8m1x3(b_u8, g_u8, r_u8);
    __riscv_vssseg3e8_v_u8m1x3(outp, stride, bgr, vl);
  } else if constexpr (b_idx == 2 && dcn == 4) {
    // RGB + Alpha
    vuint8m1_t alpha = __riscv_vmv_v_x_u8m1(0xFF, vl);
    vuint8m1x4_t rgba = __riscv_vcreate_v_u8m1x4(r_u8, g_u8, b_u8, alpha);
    __riscv_vssseg4e8_v_u8m1x4(outp, stride, rgba, vl);
  } else {
    // RGB
    vuint8m1x3_t rgb = __riscv_vcreate_v_u8m1x3(r_u8, g_u8, b_u8);
    __riscv_vssseg3e8_v_u8m1x3(outp, stride, rgb, vl);
  }
}

template <size_t b_idx, size_t u_chroma_idx, size_t y_idx, size_t dcn>
static kleidicv_error_t yuv422_to_rgbx_rvv(const uint8_t *src,
                                            size_t src_stride, uint8_t *dst,
                                            size_t dst_stride, size_t width,
                                            size_t height) {
  // Derive channel indices within each 4-byte group for vlseg4 output.
  //
  // YUYV (y_idx=0, u_chroma_idx=1): bytes [Y0,U,Y1,V]
  //   seg4 channels: ch0=Y0, ch1=U, ch2=Y1, ch3=V
  //   Y_even_ch=0, U_ch=1, Y_odd_ch=2, V_ch=3
  //
  // UYVY (y_idx=1, u_chroma_idx=0): bytes [U,Y0,V,Y1]
  //   seg4 channels: ch0=U, ch1=Y0, ch2=V, ch3=Y1
  //   Y_even_ch=1, U_ch=0, Y_odd_ch=3, V_ch=2
  //
  // YVYU (y_idx=0, u_chroma_idx=3): bytes [Y0,V,Y1,U]
  //   seg4 channels: ch0=Y0, ch1=V, ch2=Y1, ch3=U
  //   Y_even_ch=0, U_ch=3, Y_odd_ch=2, V_ch=1
  constexpr size_t Y_even_ch = y_idx;
  constexpr size_t Y_odd_ch = y_idx + 2;
  constexpr size_t U_ch = u_chroma_idx;
  constexpr size_t V_ch = (u_chroma_idx + 2) % 4;

  constexpr int32_t kRound = 1 << (kWeightScale - 1);
  constexpr ptrdiff_t out_stride = static_cast<ptrdiff_t>(2 * dcn);

  for (size_t h = 0; h < height; ++h) {
    const uint8_t *src_row = src + h * src_stride;
    uint8_t *dst_row = dst + h * dst_stride;

    size_t pairs = width / 2;  // number of pixel pairs
    size_t p = 0;              // pairs processed so far

    while (p < pairs) {
      size_t vl = __riscv_vsetvl_e8m1(pairs - p);

      // 1. Segment load: deinterleave N groups of 4 bytes into 4 channels.
      vuint8m1x4_t seg4 = __riscv_vlseg4e8_v_u8m1x4(
          src_row + p * 4, vl);

      // 2. Extract Y_even, Y_odd, U, V based on format.
      vuint8m1_t y_even_u8 = __riscv_vget_v_u8m1x4_u8m1(seg4, Y_even_ch);
      vuint8m1_t y_odd_u8  = __riscv_vget_v_u8m1x4_u8m1(seg4, Y_odd_ch);
      vuint8m1_t vu_u8     = __riscv_vget_v_u8m1x4_u8m1(seg4, U_ch);
      vuint8m1_t vv_u8     = __riscv_vget_v_u8m1x4_u8m1(seg4, V_ch);

      // 3. Widen U, V to i32: u8m1 -> u16m2 -> u32m4 -> i32m4
      vuint16m2_t vu_u16 = __riscv_vzext_vf2_u16m2(vu_u8, vl);
      vuint16m2_t vv_u16 = __riscv_vzext_vf2_u16m2(vv_u8, vl);
      vuint32m4_t vu_u32 = __riscv_vzext_vf2_u32m4(vu_u16, vl);
      vuint32m4_t vv_u32 = __riscv_vzext_vf2_u32m4(vv_u16, vl);
      vint32m4_t u32 = __riscv_vreinterpret_v_u32m4_i32m4(vu_u32);
      vint32m4_t v32 = __riscv_vreinterpret_v_u32m4_i32m4(vv_u32);

      // U' = U - 128, V' = V - 128
      vint32m4_t u_off = __riscv_vsub_vx_i32m4(u32, 128, vl);
      vint32m4_t v_off = __riscv_vsub_vx_i32m4(v32, 128, vl);

      // 4. Compute colour offsets (shared between even and odd Y pixels):
      // r_sub_y = kRV * V' + round
      vint32m4_t r_sub_y = __riscv_vmul_vx_i32m4(
          v_off, kUVWeights[kRVWeightIndex], vl);
      r_sub_y = __riscv_vadd_vx_i32m4(r_sub_y, kRound, vl);

      // g_sub_y = kGU * U' + kGV * V' + round
      vint32m4_t g_sub_y = __riscv_vmul_vx_i32m4(
          u_off, kUVWeights[kGUWeightIndex], vl);
      g_sub_y = __riscv_vmacc_vx_i32m4(
          g_sub_y, kUVWeights[kGVWeightIndex], v_off, vl);
      g_sub_y = __riscv_vadd_vx_i32m4(g_sub_y, kRound, vl);

      // b_sub_y = kBU * U' + round
      vint32m4_t b_sub_y = __riscv_vmul_vx_i32m4(
          u_off, kUVWeights[kBUWeightIndex], vl);
      b_sub_y = __riscv_vadd_vx_i32m4(b_sub_y, kRound, vl);

      // 5. Process even pixels (Y0) with strided store at even positions.
      //    Output pointer for pair p: dst_row + (p*2) * dcn
      //    Stride between successive even pixels: 2 * dcn bytes
      yuv422_row_store_rvv<b_idx, dcn>(
          y_even_u8, dst_row + p * 2 * dcn, out_stride, vl,
          r_sub_y, g_sub_y, b_sub_y);

      // 6. Process odd pixels (Y1) with strided store at odd positions.
      //    Output pointer: dst_row + (p*2 + 1) * dcn
      yuv422_row_store_rvv<b_idx, dcn>(
          y_odd_u8, dst_row + (p * 2 + 1) * dcn, out_stride, vl,
          r_sub_y, g_sub_y, b_sub_y);

      p += vl;
    }
  }
  return KLEIDICV_OK;
}

#endif  // __riscv_vector

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t yuv422_to_rgb_u8(const uint8_t *src, size_t src_stride,
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
    case KLEIDICV_YUYV_TO_BGR:
#ifdef __riscv_vector
      return yuv422_to_rgbx_rvv<0, 1, 0, 3>(src, src_stride, dst,
                                              dst_stride, width, height);
#else
      return yuv422_to_rgbx_scalar<0, 1, 0, 3>(src, src_stride, dst,
                                                 dst_stride, width, height);
#endif
    case KLEIDICV_UYVY_TO_BGR:
#ifdef __riscv_vector
      return yuv422_to_rgbx_rvv<0, 0, 1, 3>(src, src_stride, dst,
                                              dst_stride, width, height);
#else
      return yuv422_to_rgbx_scalar<0, 0, 1, 3>(src, src_stride, dst,
                                                 dst_stride, width, height);
#endif
    case KLEIDICV_YVYU_TO_BGR:
#ifdef __riscv_vector
      return yuv422_to_rgbx_rvv<0, 3, 0, 3>(src, src_stride, dst,
                                              dst_stride, width, height);
#else
      return yuv422_to_rgbx_scalar<0, 3, 0, 3>(src, src_stride, dst,
                                                 dst_stride, width, height);
#endif
    case KLEIDICV_YUYV_TO_RGB:
#ifdef __riscv_vector
      return yuv422_to_rgbx_rvv<2, 1, 0, 3>(src, src_stride, dst,
                                              dst_stride, width, height);
#else
      return yuv422_to_rgbx_scalar<2, 1, 0, 3>(src, src_stride, dst,
                                                 dst_stride, width, height);
#endif
    case KLEIDICV_UYVY_TO_RGB:
#ifdef __riscv_vector
      return yuv422_to_rgbx_rvv<2, 0, 1, 3>(src, src_stride, dst,
                                              dst_stride, width, height);
#else
      return yuv422_to_rgbx_scalar<2, 0, 1, 3>(src, src_stride, dst,
                                                 dst_stride, width, height);
#endif
    case KLEIDICV_YVYU_TO_RGB:
#ifdef __riscv_vector
      return yuv422_to_rgbx_rvv<2, 3, 0, 3>(src, src_stride, dst,
                                              dst_stride, width, height);
#else
      return yuv422_to_rgbx_scalar<2, 3, 0, 3>(src, src_stride, dst,
                                                 dst_stride, width, height);
#endif
    case KLEIDICV_YUYV_TO_BGRA:
#ifdef __riscv_vector
      return yuv422_to_rgbx_rvv<0, 1, 0, 4>(src, src_stride, dst,
                                              dst_stride, width, height);
#else
      return yuv422_to_rgbx_scalar<0, 1, 0, 4>(src, src_stride, dst,
                                                 dst_stride, width, height);
#endif
    case KLEIDICV_UYVY_TO_BGRA:
#ifdef __riscv_vector
      return yuv422_to_rgbx_rvv<0, 0, 1, 4>(src, src_stride, dst,
                                              dst_stride, width, height);
#else
      return yuv422_to_rgbx_scalar<0, 0, 1, 4>(src, src_stride, dst,
                                                 dst_stride, width, height);
#endif
    case KLEIDICV_YVYU_TO_BGRA:
#ifdef __riscv_vector
      return yuv422_to_rgbx_rvv<0, 3, 0, 4>(src, src_stride, dst,
                                              dst_stride, width, height);
#else
      return yuv422_to_rgbx_scalar<0, 3, 0, 4>(src, src_stride, dst,
                                                 dst_stride, width, height);
#endif
    case KLEIDICV_YUYV_TO_RGBA:
#ifdef __riscv_vector
      return yuv422_to_rgbx_rvv<2, 1, 0, 4>(src, src_stride, dst,
                                              dst_stride, width, height);
#else
      return yuv422_to_rgbx_scalar<2, 1, 0, 4>(src, src_stride, dst,
                                                 dst_stride, width, height);
#endif
    case KLEIDICV_UYVY_TO_RGBA:
#ifdef __riscv_vector
      return yuv422_to_rgbx_rvv<2, 0, 1, 4>(src, src_stride, dst,
                                              dst_stride, width, height);
#else
      return yuv422_to_rgbx_scalar<2, 0, 1, 4>(src, src_stride, dst,
                                                 dst_stride, width, height);
#endif
    case KLEIDICV_YVYU_TO_RGBA:
#ifdef __riscv_vector
      return yuv422_to_rgbx_rvv<2, 3, 0, 4>(src, src_stride, dst,
                                              dst_stride, width, height);
#else
      return yuv422_to_rgbx_scalar<2, 3, 0, 4>(src, src_stride, dst,
                                                 dst_stride, width, height);
#endif
    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
}

}  // namespace kleidicv::neon
