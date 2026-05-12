// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>

#include "kleidicv/arithmetics/scale.h"
#include "kleidicv/kleidicv.h"

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

#ifdef __riscv_vector

// --- uint8 scale helpers ---

static inline uint8_t scale_value_u8(uint8_t value, float scale, float shift) {
  int64_t v = lrintf(static_cast<float>(value) * scale + shift);
  if (v < 0) return 0;
  if (v > 255) return 255;
  return static_cast<uint8_t>(v);
}

// --- float scale helpers ---

static inline float scale_value_f32(float value, float scale, float shift) {
  return std::fma(value, scale, shift);
}

#endif  // __riscv_vector

// Precalculate scale table for uint8 (scalar implementation for RVV builds).
std::array<uint8_t, 256> precalculate_scale_table_u8(double dscale,
                                                     double dshift) {
  float scale = static_cast<float>(dscale);
  float shift = static_cast<float>(dshift);
  std::array<uint8_t, 256> table{};
  for (size_t i = 0; i < 256; ++i) {
    int64_t v = lrintf(static_cast<float>(i) * scale + shift);
    if (v < 0) v = 0;
    if (v > 255) v = 255;
    table[i] = static_cast<uint8_t>(v);
  }
  return table;
}

kleidicv_error_t scale_with_precalculated_table_u8(
    const uint8_t *src, size_t src_stride, uint8_t *dst, size_t dst_stride,
    size_t width, size_t height, double scale, double shift,
    const std::array<uint8_t, 256> &precalculated_table) {
  (void)scale;
  (void)shift;

  for (size_t row = 0; row < height; ++row) {
    const uint8_t *s = reinterpret_cast<const uint8_t *>(
        reinterpret_cast<const uint8_t *>(src) + row * src_stride);
    uint8_t *d = reinterpret_cast<uint8_t *>(reinterpret_cast<uint8_t *>(dst) +
                                             row * dst_stride);
    for (size_t i = 0; i < width; ++i) {
      d[i] = precalculated_table[s[i]];
    }
  }

  return KLEIDICV_OK;
}

// Specialization for uint8_t to uint8_t
template <>
kleidicv_error_t scale(const uint8_t *src, size_t src_stride, uint8_t *dst,
                       size_t dst_stride, size_t width, size_t height,
                       double scale, double shift) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

#ifdef __riscv_vector
  // For uint8, use the precalculated table approach (fast lookup).
  auto precalculated_table = precalculate_scale_table_u8(scale, shift);
  return scale_with_precalculated_table_u8(src, src_stride, dst, dst_stride,
                                           width, height, scale, shift,
                                           precalculated_table);
#else
  auto precalculated_table = precalculate_scale_table_u8(scale, shift);
  return scale_with_precalculated_table_u8(src, src_stride, dst, dst_stride,
                                           width, height, scale, shift,
                                           precalculated_table);
#endif
}

// Specialization for float to float
template <>
kleidicv_error_t scale(const float *src, size_t src_stride, float *dst,
                       size_t dst_stride, size_t width, size_t height,
                       double scale, double shift) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  float fscale = static_cast<float>(scale);
  float fshift = static_cast<float>(shift);

#ifdef __riscv_vector
  for (size_t row = 0; row < height; ++row) {
    const float *s = row_ptr(src, src_stride, row);
    float *d = row_ptr(dst, dst_stride, row);
    size_t i = 0;
    while (i < width) {
      size_t vl = rvv_setvl<float>(width - i);
      vfloat32m1_t vs = rvv_load(s + i, vl);
      vfloat32m1_t vsc = rvv_splat_f32(fscale, vl);
      vfloat32m1_t vsh = rvv_splat_f32(fshift, vl);
      // dst = src * scale + shift
      vfloat32m1_t vr = __riscv_vfmadd_vv_f32m1(vs, vsc, vsh, vl);
      rvv_store(d + i, vr, vl);
      i += vl;
    }
  }
#else
  for (size_t row = 0; row < height; ++row) {
    const float *s = reinterpret_cast<const float *>(
        reinterpret_cast<const uint8_t *>(src) + row * src_stride);
    float *d = reinterpret_cast<float *>(reinterpret_cast<uint8_t *>(dst) +
                                         row * dst_stride);
    for (size_t i = 0; i < width; ++i) {
      d[i] = std::fma(s[i], fscale, fshift);
    }
  }
#endif

  return KLEIDICV_OK;
}

// Specialization for uint8_t to float16_t
// Use scalar fallback since float16 support via RVV requires Zvfh extension.
template <>
kleidicv_error_t scale(const uint8_t *src, size_t src_stride, float16_t *dst,
                       size_t dst_stride, size_t width, size_t height,
                       double scale, double shift) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  float fscale = static_cast<float>(scale);
  float fshift = static_cast<float>(shift);

  for (size_t row = 0; row < height; ++row) {
    const uint8_t *s = reinterpret_cast<const uint8_t *>(
        reinterpret_cast<const uint8_t *>(src) + row * src_stride);
    float16_t *d = reinterpret_cast<float16_t *>(
        reinterpret_cast<uint8_t *>(dst) + row * dst_stride);
    for (size_t i = 0; i < width; ++i) {
      d[i] = static_cast<float16_t>(static_cast<float>(s[i]) * fscale + fshift);
    }
  }

  return KLEIDICV_OK;
}

}  // namespace kleidicv::neon
