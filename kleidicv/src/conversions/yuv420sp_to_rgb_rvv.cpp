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

// Scalar YUV420sp (NV12/NV21) to RGB/BGR/RGBA/BGRA conversion.

static inline uint8_t clamp_u8(int32_t v) {
  return static_cast<uint8_t>(std::clamp(v, 0, 255));
}

template <bool BGR, bool kAlpha>
static kleidicv_error_t yuv420sp_to_rgbx_scalar(
    const uint8_t *src_y, size_t src_y_stride, const uint8_t *src_uv,
    size_t src_uv_stride, uint8_t *dst, size_t dst_stride, size_t width,
    size_t height, bool v_first) {
  constexpr size_t dcn = kAlpha ? 4 : 3;

  for (size_t h = 0; h < height; h += 2) {
    const uint8_t *y0 = src_y + h * src_y_stride;
    const uint8_t *y1 = y0 + src_y_stride;
    const uint8_t *uv = src_uv + (h / 2) * src_uv_stride;
    uint8_t *row0 = dst + h * dst_stride;
    uint8_t *row1 = dst + (h + 1) * dst_stride;

    if (h + 1 >= height) {
      y1 = y0;
      row1 = row0;
    }

    for (size_t x = 0; x < width; ++x) {
      int32_t u_val = uv[(x / 2) * 2 + 0];
      int32_t v_val = uv[(x / 2) * 2 + 1];
      if (v_first) std::swap(u_val, v_val);

      int32_t u_m128 = u_val - 128;
      int32_t v_m128 = v_val - 128;

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
  }
  return KLEIDICV_OK;
}

#ifdef __riscv_vector

// RVV vectorized YUV420sp (NV12/NV21) to RGB/BGR/RGBA/BGRA conversion.
//
// The UV plane is semi-planar: interleaved U,V byte pairs at half resolution
// in both horizontal and vertical directions.  Each UV pair covers a 2x2 block
// of Y pixels.
//
// Strategy per row:
//   1. Set vl for e8m1 on the remaining Y pixels, round down to even so that
//      every UV pair maps to exactly 2 Y pixels within the vector.
//   2. Segment-load vl/2 interleaved UV pairs with vlseg2e8 to get separate
//      U and V vectors of length vl/2.
//   3. Duplicate each U,V value to cover 2 Y pixels using a gather index
//      [0,0,1,1,2,2,...] built from (vid >> 1).
//   4. Widen U,V to i32, subtract 128, compute colour offsets (r_sub_y, etc.).
//   5. Load vl Y bytes, saturating-subtract 16, widen to i32, multiply by
//      kYWeight, add colour offsets and rounding bias, arithmetic-shift right
//      by kWeightScale (20).
//   6. Clamp to [0,255], narrow i32 -> u16 -> u8, segment-store RGB(A).
//   7. After the even-vl loop, process any remaining 1 pixel with scalar code.
//
// Both rows of the pair (y0,y1) reuse the same UV data.
// VLEN-agnostic: vl is obtained from vsetvl each iteration.

// Helper: process one row of Y pixels against precomputed UV colour offsets.
//
// r_sub_y, g_sub_y, b_sub_y are i32m4 vectors of length vl containing:
//   r_sub_y[i] = kRV * (V'[i]) + round
//   g_sub_y[i] = kGU * (U'[i]) + kGV * (V'[i]) + round
//   b_sub_y[i] = kBU * (U'[i]) + round
// where U' = U-128, V' = V-128, round = 1 << (kWeightScale-1).
template <bool BGR, bool kAlpha>
static inline void yuv420sp_row_rvv(
    const uint8_t *yp, uint8_t *outp, size_t vl,
    vint32m4_t r_sub_y, vint32m4_t g_sub_y, vint32m4_t b_sub_y) {
  // Load Y bytes and compute Y' = max(Y - 16, 0)
  vuint8m1_t vy_u8 = __riscv_vle8_v_u8m1(yp, vl);
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

  // Segment store with channel order depending on BGR/RGB and alpha
  if constexpr (BGR && kAlpha) {
    vuint8m1_t alpha = __riscv_vmv_v_x_u8m1(0xFF, vl);
    vuint8m1x4_t bgra = __riscv_vcreate_v_u8m1x4(b_u8, g_u8, r_u8, alpha);
    __riscv_vsseg4e8_v_u8m1x4(outp, bgra, vl);
  } else if constexpr (BGR && !kAlpha) {
    vuint8m1x3_t bgr = __riscv_vcreate_v_u8m1x3(b_u8, g_u8, r_u8);
    __riscv_vsseg3e8_v_u8m1x3(outp, bgr, vl);
  } else if constexpr (!BGR && kAlpha) {
    vuint8m1_t alpha = __riscv_vmv_v_x_u8m1(0xFF, vl);
    vuint8m1x4_t rgba = __riscv_vcreate_v_u8m1x4(r_u8, g_u8, b_u8, alpha);
    __riscv_vsseg4e8_v_u8m1x4(outp, rgba, vl);
  } else {
    vuint8m1x3_t rgb = __riscv_vcreate_v_u8m1x3(r_u8, g_u8, b_u8);
    __riscv_vsseg3e8_v_u8m1x3(outp, rgb, vl);
  }
}

template <bool BGR, bool kAlpha>
static kleidicv_error_t yuv420sp_to_rgbx_rvv(
    const uint8_t *src_y, size_t src_y_stride, const uint8_t *src_uv,
    size_t src_uv_stride, uint8_t *dst, size_t dst_stride, size_t width,
    size_t height, bool v_first) {
  constexpr size_t dcn = kAlpha ? 4 : 3;
  constexpr int32_t kRound = 1 << (kWeightScale - 1);

  for (size_t h = 0; h < height; h += 2) {
    const uint8_t *y0 = src_y + h * src_y_stride;
    const uint8_t *y1 = y0 + src_y_stride;
    const uint8_t *uv = src_uv + (h / 2) * src_uv_stride;
    uint8_t *row0 = dst + h * dst_stride;
    uint8_t *row1 = dst + (h + 1) * dst_stride;

    if (h + 1 >= height) {
      y1 = y0;
      row1 = row0;
    }

    size_t x = 0;

    // Vectorized loop: process an even number of pixels per iteration.
    // Each UV pair covers 2 horizontal Y pixels, so we round vl down to even.
    while (x + 1 < width) {
      // Determine how many Y pixels to process this iteration (at least 2).
      size_t avl = width - x;
      size_t vl = __riscv_vsetvl_e8m1(avl);
      // Round down to even so UV pairs map cleanly to Y pixel pairs.
      vl = vl & ~static_cast<size_t>(1);
      if (vl == 0) break;  // remaining 1 pixel handled by scalar tail

      size_t uv_count = vl / 2;  // number of UV pairs

      // Load interleaved UV pairs: deinterleave into separate U and V vectors.
      // UV plane layout: [U0,V0, U1,V1, U2,V2, ...]
      // After vlseg2e8 with uv_count: ch0 = [U0,U1,...], ch1 = [V0,V1,...]
      vuint8mf2x2_t uv_pair =
          __riscv_vlseg2e8_v_u8mf2x2(uv + x, uv_count);
      vuint8mf2_t u_half = __riscv_vget_v_u8mf2x2_u8mf2(uv_pair, 0);
      vuint8mf2_t v_half = __riscv_vget_v_u8mf2x2_u8mf2(uv_pair, 1);

      if (v_first) {
        vuint8mf2_t tmp = u_half;
        u_half = v_half;
        v_half = tmp;
      }

      // Duplicate each UV value for 2 Y pixels using vrgatherei16 with
      // index [0,0,1,1,2,2,...].  Build the index from vid >> 1.
      // vrgatherei16 for u8m1 data requires u16m2 index (EEW ratio = 2).
      vuint16m2_t vid = __riscv_vid_v_u16m2(vl);
      vuint16m2_t dup_idx = __riscv_vsrl_vx_u16m2(vid, 1, vl);

      // Widen u_half (mf2, uv_count elems) to u8m1 via vzext, then gather
      // Actually, we need to gather from mf2 vectors.  Let's widen first
      // then gather at u8m1 level.
      // Approach: widen mf2 -> m1 is not a standard op. Instead, use
      // vlmul_ext to reinterpret mf2 as m1 (upper half undefined but we
      // only gather within uv_count range).
      vuint8m1_t u_ext = __riscv_vlmul_ext_v_u8mf2_u8m1(u_half);
      vuint8m1_t v_ext = __riscv_vlmul_ext_v_u8mf2_u8m1(v_half);

      // Gather with 16-bit indices to produce duplicated u8m1 vectors of
      // length vl: element i reads from index i/2.
      vuint8m1_t u_dup = __riscv_vrgatherei16_vv_u8m1(u_ext, dup_idx, vl);
      vuint8m1_t v_dup = __riscv_vrgatherei16_vv_u8m1(v_ext, dup_idx, vl);

      // Widen U, V to i32: u8m1 -> u16m2 -> u32m4 -> i32m4
      vuint16m2_t u_u16 = __riscv_vzext_vf2_u16m2(u_dup, vl);
      vuint16m2_t v_u16 = __riscv_vzext_vf2_u16m2(v_dup, vl);
      vuint32m4_t u_u32 = __riscv_vzext_vf2_u32m4(u_u16, vl);
      vuint32m4_t v_u32 = __riscv_vzext_vf2_u32m4(v_u16, vl);
      vint32m4_t u32 = __riscv_vreinterpret_v_u32m4_i32m4(u_u32);
      vint32m4_t v32 = __riscv_vreinterpret_v_u32m4_i32m4(v_u32);

      // U' = U - 128, V' = V - 128
      vint32m4_t u_off = __riscv_vsub_vx_i32m4(u32, 128, vl);
      vint32m4_t v_off = __riscv_vsub_vx_i32m4(v32, 128, vl);

      // Compute colour offsets (shared between row0 and row1):
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

      // Process row 0
      yuv420sp_row_rvv<BGR, kAlpha>(
          y0 + x, row0 + x * dcn, vl, r_sub_y, g_sub_y, b_sub_y);

      // Process row 1 (reuses UV colour offsets)
      yuv420sp_row_rvv<BGR, kAlpha>(
          y1 + x, row1 + x * dcn, vl, r_sub_y, g_sub_y, b_sub_y);

      x += vl;
    }

    // Scalar tail: handle remaining pixel (at most 1 when width is odd).
    for (; x < width; ++x) {
      int32_t u_val = uv[(x / 2) * 2 + 0];
      int32_t v_val = uv[(x / 2) * 2 + 1];
      if (v_first) std::swap(u_val, v_val);

      int32_t u_m128 = u_val - 128;
      int32_t v_m128 = v_val - 128;

      for (int sel = 0; sel < 2; ++sel) {
        const uint8_t *yp = (sel == 0) ? y0 : y1;
        uint8_t *outp = (sel == 0) ? row0 : row1;

        int32_t yv = kYWeight * std::max(static_cast<int>(yp[x]) - 16, 0);
        int32_t r = yv + kUVWeights[kRVWeightIndex] * v_m128;
        int32_t g = yv + kUVWeights[kGUWeightIndex] * u_m128 +
                    kUVWeights[kGVWeightIndex] * v_m128;
        int32_t b = yv + kUVWeights[kBUWeightIndex] * u_m128;

        r = (r + kRound) >> kWeightScale;
        g = (g + kRound) >> kWeightScale;
        b = (b + kRound) >> kWeightScale;

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
  }
  return KLEIDICV_OK;
}

#endif  // __riscv_vector

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t yuv420sp_to_rgb_u8(const uint8_t *src_y, size_t src_y_stride,
                                    const uint8_t *src_uv, size_t src_uv_stride,
                                    uint8_t *dst, size_t dst_stride,
                                    size_t width, size_t height,
                                    kleidicv_color_conversion_t color_format) {
  CHECK_POINTER_AND_STRIDE(src_y, src_y_stride, height);
  CHECK_POINTER_AND_STRIDE(src_uv, src_uv_stride, (height + 1) / 2);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  switch (color_format) {
    case KLEIDICV_NV21_TO_BGR:
#ifdef __riscv_vector
      return yuv420sp_to_rgbx_rvv<true, false>(
          src_y, src_y_stride, src_uv, src_uv_stride, dst, dst_stride, width,
          height, true);
#else
      return yuv420sp_to_rgbx_scalar<true, false>(
          src_y, src_y_stride, src_uv, src_uv_stride, dst, dst_stride, width,
          height, true);
#endif
    case KLEIDICV_NV21_TO_RGB:
#ifdef __riscv_vector
      return yuv420sp_to_rgbx_rvv<false, false>(
          src_y, src_y_stride, src_uv, src_uv_stride, dst, dst_stride, width,
          height, true);
#else
      return yuv420sp_to_rgbx_scalar<false, false>(
          src_y, src_y_stride, src_uv, src_uv_stride, dst, dst_stride, width,
          height, true);
#endif
    case KLEIDICV_NV21_TO_BGRA:
#ifdef __riscv_vector
      return yuv420sp_to_rgbx_rvv<true, true>(
          src_y, src_y_stride, src_uv, src_uv_stride, dst, dst_stride, width,
          height, true);
#else
      return yuv420sp_to_rgbx_scalar<true, true>(
          src_y, src_y_stride, src_uv, src_uv_stride, dst, dst_stride, width,
          height, true);
#endif
    case KLEIDICV_NV21_TO_RGBA:
#ifdef __riscv_vector
      return yuv420sp_to_rgbx_rvv<false, true>(
          src_y, src_y_stride, src_uv, src_uv_stride, dst, dst_stride, width,
          height, true);
#else
      return yuv420sp_to_rgbx_scalar<false, true>(
          src_y, src_y_stride, src_uv, src_uv_stride, dst, dst_stride, width,
          height, true);
#endif
    case KLEIDICV_NV12_TO_BGR:
#ifdef __riscv_vector
      return yuv420sp_to_rgbx_rvv<true, false>(
          src_y, src_y_stride, src_uv, src_uv_stride, dst, dst_stride, width,
          height, false);
#else
      return yuv420sp_to_rgbx_scalar<true, false>(
          src_y, src_y_stride, src_uv, src_uv_stride, dst, dst_stride, width,
          height, false);
#endif
    case KLEIDICV_NV12_TO_RGB:
#ifdef __riscv_vector
      return yuv420sp_to_rgbx_rvv<false, false>(
          src_y, src_y_stride, src_uv, src_uv_stride, dst, dst_stride, width,
          height, false);
#else
      return yuv420sp_to_rgbx_scalar<false, false>(
          src_y, src_y_stride, src_uv, src_uv_stride, dst, dst_stride, width,
          height, false);
#endif
    case KLEIDICV_NV12_TO_BGRA:
#ifdef __riscv_vector
      return yuv420sp_to_rgbx_rvv<true, true>(
          src_y, src_y_stride, src_uv, src_uv_stride, dst, dst_stride, width,
          height, false);
#else
      return yuv420sp_to_rgbx_scalar<true, true>(
          src_y, src_y_stride, src_uv, src_uv_stride, dst, dst_stride, width,
          height, false);
#endif
    case KLEIDICV_NV12_TO_RGBA:
#ifdef __riscv_vector
      return yuv420sp_to_rgbx_rvv<false, true>(
          src_y, src_y_stride, src_uv, src_uv_stride, dst, dst_stride, width,
          height, false);
#else
      return yuv420sp_to_rgbx_scalar<false, true>(
          src_y, src_y_stride, src_uv, src_uv_stride, dst, dst_stride, width,
          height, false);
#endif
    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
}

}  // namespace kleidicv::neon
