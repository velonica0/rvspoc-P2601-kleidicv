// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <limits>

#include "kleidicv/conversions/float_conversion.h"

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

#ifdef __riscv_vector

// --- f32 -> u8 ---

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t f32_to_u8(const float *src, size_t src_stride, uint8_t *dst,
                           size_t dst_stride, size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  for (size_t y = 0; y < height; ++y) {
    const float *src_row = row_ptr(src, src_stride, y);
    uint8_t *dst_row = row_ptr(dst, dst_stride, y);

    size_t x = 0;
    while (x < width) {
      size_t vl = __riscv_vsetvl_e32m4(width - x);
      vfloat32m4_t vf = __riscv_vle32_v_f32m4(src_row + x, vl);
      // Replace NaN with 0 (RISC-V vfcvt maps NaN to INT_MAX, ARM maps to 0)
      vbool8_t nan_mask = __riscv_vmfne_vv_f32m4_b8(vf, vf, vl);
      vf = __riscv_vfmerge_vfm_f32m4(vf, 0.0f, nan_mask, vl);
      vuint32m4_t vi = __riscv_vfcvt_xu_f_v_u32m4(vf, vl);
      vuint16m2_t v16 = __riscv_vnclipu_wx_u16m2(vi, 0, __RISCV_VXRM_RNU, vl);
      vuint8m1_t v8 = __riscv_vnclipu_wx_u8m1(v16, 0, __RISCV_VXRM_RNU, vl);
      __riscv_vse8_v_u8m1(dst_row + x, v8, vl);
      x += vl;
    }
  }

  return KLEIDICV_OK;
}

// --- f32 -> s8 ---

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t f32_to_s8(const float *src, size_t src_stride, int8_t *dst,
                           size_t dst_stride, size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  for (size_t y = 0; y < height; ++y) {
    const float *src_row = row_ptr(src, src_stride, y);
    int8_t *dst_row = row_ptr(dst, dst_stride, y);

    size_t x = 0;
    while (x < width) {
      size_t vl = __riscv_vsetvl_e32m4(width - x);
      vfloat32m4_t vf = __riscv_vle32_v_f32m4(src_row + x, vl);
      // Replace NaN with 0 (RISC-V vfcvt maps NaN to INT_MAX, ARM maps to 0)
      vbool8_t nan_mask = __riscv_vmfne_vv_f32m4_b8(vf, vf, vl);
      vf = __riscv_vfmerge_vfm_f32m4(vf, 0.0f, nan_mask, vl);
      vint32m4_t vi = __riscv_vfcvt_x_f_v_i32m4(vf, vl);
      vint16m2_t v16 = __riscv_vnclip_wx_i16m2(vi, 0, __RISCV_VXRM_RNU, vl);
      vint8m1_t v8 = __riscv_vnclip_wx_i8m1(v16, 0, __RISCV_VXRM_RNU, vl);
      __riscv_vse8_v_i8m1(dst_row + x, v8, vl);
      x += vl;
    }
  }

  return KLEIDICV_OK;
}

// --- s8 -> f32 ---

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
s8_to_f32(const int8_t *src, size_t src_stride, float *dst, size_t dst_stride,
          size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  for (size_t y = 0; y < height; ++y) {
    const int8_t *src_row = row_ptr(src, src_stride, y);
    float *dst_row = row_ptr(dst, dst_stride, y);

    size_t x = 0;
    while (x < width) {
      size_t vl = __riscv_vsetvl_e8m1(width - x);
      // Load int8
      vint8m1_t v8 = __riscv_vle8_v_i8m1(src_row + x, vl);
      // Widen i8 -> i16
      vint16m2_t v16 = __riscv_vsext_vf2_i16m2(v8, vl);
      // Widen i16 -> i32
      vint32m4_t v32 = __riscv_vsext_vf2_i32m4(v16, vl);
      // Convert i32 -> f32
      vfloat32m4_t vf = __riscv_vfcvt_f_x_v_f32m4(v32, vl);
      // Store
      __riscv_vse32_v_f32m4(dst_row + x, vf, vl);
      x += vl;
    }
  }

  return KLEIDICV_OK;
}

// --- u8 -> f32 ---

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
u8_to_f32(const uint8_t *src, size_t src_stride, float *dst, size_t dst_stride,
          size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *src_row = row_ptr(src, src_stride, y);
    float *dst_row = row_ptr(dst, dst_stride, y);

    size_t x = 0;
    while (x < width) {
      size_t vl = __riscv_vsetvl_e8m1(width - x);
      // Load uint8
      vuint8m1_t v8 = __riscv_vle8_v_u8m1(src_row + x, vl);
      // Widen u8 -> u16
      vuint16m2_t v16 = __riscv_vzext_vf2_u16m2(v8, vl);
      // Widen u16 -> u32
      vuint32m4_t v32 = __riscv_vzext_vf2_u32m4(v16, vl);
      // Convert u32 -> f32
      vfloat32m4_t vf = __riscv_vfcvt_f_xu_v_f32m4(v32, vl);
      // Store
      __riscv_vse32_v_f32m4(dst_row + x, vf, vl);
      x += vl;
    }
  }

  return KLEIDICV_OK;
}

#else  // scalar fallback

kleidicv_error_t f32_to_u8(const float *src, size_t src_stride, uint8_t *dst,
                           size_t dst_stride, size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  for (size_t y = 0; y < height; ++y) {
    const float *s = reinterpret_cast<const float *>(
        reinterpret_cast<const uint8_t *>(src) + y * src_stride);
    uint8_t *d = reinterpret_cast<uint8_t *>(
        reinterpret_cast<uint8_t *>(dst) + y * dst_stride);
    for (size_t x = 0; x < width; ++x) {
      float f = s[x];
      if (std::isnan(f)) { d[x] = 0; continue; }
      f = std::nearbyint(f);
      if (f > 255.0f) f = 255.0f;
      if (f < 0.0f) f = 0.0f;
      d[x] = static_cast<uint8_t>(f);
    }
  }
  return KLEIDICV_OK;
}

kleidicv_error_t f32_to_s8(const float *src, size_t src_stride, int8_t *dst,
                           size_t dst_stride, size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  for (size_t y = 0; y < height; ++y) {
    const float *s = reinterpret_cast<const float *>(
        reinterpret_cast<const uint8_t *>(src) + y * src_stride);
    int8_t *d = reinterpret_cast<int8_t *>(
        reinterpret_cast<uint8_t *>(dst) + y * dst_stride);
    for (size_t x = 0; x < width; ++x) {
      float f = s[x];
      if (std::isnan(f)) { d[x] = 0; continue; }
      f = std::nearbyint(f);
      if (f > 127.0f) f = 127.0f;
      if (f < -128.0f) f = -128.0f;
      d[x] = static_cast<int8_t>(f);
    }
  }
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
s8_to_f32(const int8_t *src, size_t src_stride, float *dst, size_t dst_stride,
          size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  for (size_t y = 0; y < height; ++y) {
    const int8_t *s = reinterpret_cast<const int8_t *>(
        reinterpret_cast<const uint8_t *>(src) + y * src_stride);
    float *d = reinterpret_cast<float *>(
        reinterpret_cast<uint8_t *>(dst) + y * dst_stride);
    for (size_t x = 0; x < width; ++x) {
      d[x] = static_cast<float>(s[x]);
    }
  }
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
u8_to_f32(const uint8_t *src, size_t src_stride, float *dst, size_t dst_stride,
          size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *s = reinterpret_cast<const uint8_t *>(
        reinterpret_cast<const uint8_t *>(src) + y * src_stride);
    float *d = reinterpret_cast<float *>(
        reinterpret_cast<uint8_t *>(dst) + y * dst_stride);
    for (size_t x = 0; x < width; ++x) {
      d[x] = static_cast<float>(s[x]);
    }
  }
  return KLEIDICV_OK;
}

#endif  // __riscv_vector

}  // namespace kleidicv::neon
