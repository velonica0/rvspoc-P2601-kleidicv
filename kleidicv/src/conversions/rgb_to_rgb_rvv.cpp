// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/conversions/rgb_to_rgb.h"
#include "kleidicv/kleidicv.h"

#include "kleidicv/rvv.h"

#include <cstring>

namespace kleidicv::neon {

#ifdef __riscv_vector

// --- RGB <-> BGR (3-channel swap of channels 0 and 2) ---

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgb_to_bgr_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *src_row = row_ptr(src, src_stride, y);
    uint8_t *dst_row = row_ptr(dst, dst_stride, y);

    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e8(width - x);
      vuint8m1x3_t rgb_tuple = __riscv_vlseg3e8_v_u8m1x3(src_row + x * 3, vl);
      vuint8m1_t r = __riscv_vget_v_u8m1x3_u8m1(rgb_tuple, 0);
      vuint8m1_t g = __riscv_vget_v_u8m1x3_u8m1(rgb_tuple, 1);
      vuint8m1_t b = __riscv_vget_v_u8m1x3_u8m1(rgb_tuple, 2);
      // Swap R and B channels
      vuint8m1x3_t bgr_tuple = __riscv_vcreate_v_u8m1x3(b, g, r);
      __riscv_vsseg3e8_v_u8m1x3(dst_row + x * 3, bgr_tuple, vl);
      x += vl;
    }
  }

  return KLEIDICV_OK;
}

// --- RGBA <-> BGRA (4-channel swap of channels 0 and 2) ---

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgba_to_bgra_u8(const uint8_t *src, size_t src_stride,
                                 uint8_t *dst, size_t dst_stride, size_t width,
                                 size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *src_row = row_ptr(src, src_stride, y);
    uint8_t *dst_row = row_ptr(dst, dst_stride, y);

    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e8(width - x);
      vuint8m1x4_t rgba_tuple = __riscv_vlseg4e8_v_u8m1x4(src_row + x * 4, vl);
      vuint8m1_t r = __riscv_vget_v_u8m1x4_u8m1(rgba_tuple, 0);
      vuint8m1_t g = __riscv_vget_v_u8m1x4_u8m1(rgba_tuple, 1);
      vuint8m1_t b = __riscv_vget_v_u8m1x4_u8m1(rgba_tuple, 2);
      vuint8m1_t a = __riscv_vget_v_u8m1x4_u8m1(rgba_tuple, 3);
      // Swap R and B channels, keep G and A
      vuint8m1x4_t bgra_tuple = __riscv_vcreate_v_u8m1x4(b, g, r, a);
      __riscv_vsseg4e8_v_u8m1x4(dst_row + x * 4, bgra_tuple, vl);
      x += vl;
    }
  }

  return KLEIDICV_OK;
}

// --- RGB -> BGRA (3ch to 4ch, swap R and B, add alpha=0xFF) ---

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
rgb_to_bgra_u8(const uint8_t *src, size_t src_stride, uint8_t *dst,
               size_t dst_stride, size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *src_row = row_ptr(src, src_stride, y);
    uint8_t *dst_row = row_ptr(dst, dst_stride, y);

    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e8(width - x);
      vuint8m1x3_t rgb_tuple = __riscv_vlseg3e8_v_u8m1x3(src_row + x * 3, vl);
      vuint8m1_t r = __riscv_vget_v_u8m1x3_u8m1(rgb_tuple, 0);
      vuint8m1_t g = __riscv_vget_v_u8m1x3_u8m1(rgb_tuple, 1);
      vuint8m1_t b = __riscv_vget_v_u8m1x3_u8m1(rgb_tuple, 2);
      vuint8m1_t alpha = rvv_splat(uint8_t(0xFF), vl);
      // B, G, R, A
      vuint8m1x4_t bgra_tuple = __riscv_vcreate_v_u8m1x4(b, g, r, alpha);
      __riscv_vsseg4e8_v_u8m1x4(dst_row + x * 4, bgra_tuple, vl);
      x += vl;
    }
  }

  return KLEIDICV_OK;
}

// --- RGB -> RGBA (3ch to 4ch, add alpha=0xFF) ---

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
rgb_to_rgba_u8(const uint8_t *src, size_t src_stride, uint8_t *dst,
               size_t dst_stride, size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *src_row = row_ptr(src, src_stride, y);
    uint8_t *dst_row = row_ptr(dst, dst_stride, y);

    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e8(width - x);
      vuint8m1x3_t rgb_tuple = __riscv_vlseg3e8_v_u8m1x3(src_row + x * 3, vl);
      vuint8m1_t r = __riscv_vget_v_u8m1x3_u8m1(rgb_tuple, 0);
      vuint8m1_t g = __riscv_vget_v_u8m1x3_u8m1(rgb_tuple, 1);
      vuint8m1_t b = __riscv_vget_v_u8m1x3_u8m1(rgb_tuple, 2);
      vuint8m1_t alpha = rvv_splat(uint8_t(0xFF), vl);
      vuint8m1x4_t rgba_tuple = __riscv_vcreate_v_u8m1x4(r, g, b, alpha);
      __riscv_vsseg4e8_v_u8m1x4(dst_row + x * 4, rgba_tuple, vl);
      x += vl;
    }
  }

  return KLEIDICV_OK;
}

// --- RGBA -> BGR (4ch to 3ch, swap R and B, drop alpha) ---

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgba_to_bgr_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *src_row = row_ptr(src, src_stride, y);
    uint8_t *dst_row = row_ptr(dst, dst_stride, y);

    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e8(width - x);
      vuint8m1x4_t rgba_tuple = __riscv_vlseg4e8_v_u8m1x4(src_row + x * 4, vl);
      vuint8m1_t r = __riscv_vget_v_u8m1x4_u8m1(rgba_tuple, 0);
      vuint8m1_t g = __riscv_vget_v_u8m1x4_u8m1(rgba_tuple, 1);
      vuint8m1_t b = __riscv_vget_v_u8m1x4_u8m1(rgba_tuple, 2);
      vuint8m1x3_t bgr_tuple = __riscv_vcreate_v_u8m1x3(b, g, r);
      __riscv_vsseg3e8_v_u8m1x3(dst_row + x * 3, bgr_tuple, vl);
      x += vl;
    }
  }

  return KLEIDICV_OK;
}

// --- RGBA -> RGB (4ch to 3ch, drop alpha) ---

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgba_to_rgb_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *src_row = row_ptr(src, src_stride, y);
    uint8_t *dst_row = row_ptr(dst, dst_stride, y);

    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e8(width - x);
      vuint8m1x4_t rgba_tuple = __riscv_vlseg4e8_v_u8m1x4(src_row + x * 4, vl);
      vuint8m1_t r = __riscv_vget_v_u8m1x4_u8m1(rgba_tuple, 0);
      vuint8m1_t g = __riscv_vget_v_u8m1x4_u8m1(rgba_tuple, 1);
      vuint8m1_t b = __riscv_vget_v_u8m1x4_u8m1(rgba_tuple, 2);
      vuint8m1x3_t rgb_tuple = __riscv_vcreate_v_u8m1x3(r, g, b);
      __riscv_vsseg3e8_v_u8m1x3(dst_row + x * 3, rgb_tuple, vl);
      x += vl;
    }
  }

  return KLEIDICV_OK;
}

#else  // scalar fallback

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgb_to_bgr_u8(const uint8_t *src, size_t src_stride,
                               uint8_t *dst, size_t dst_stride, size_t width,
                               size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *s = reinterpret_cast<const uint8_t *>(
        reinterpret_cast<const uint8_t *>(src) + y * src_stride);
    uint8_t *d = reinterpret_cast<uint8_t *>(
        reinterpret_cast<uint8_t *>(dst) + y * dst_stride);
    for (size_t x = 0; x < width; ++x) {
      uint8_t r = s[x * 3 + 0], g = s[x * 3 + 1], b = s[x * 3 + 2];
      d[x * 3 + 0] = b;
      d[x * 3 + 1] = g;
      d[x * 3 + 2] = r;
    }
  }
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgba_to_bgra_u8(const uint8_t *src, size_t src_stride,
                                 uint8_t *dst, size_t dst_stride, size_t width,
                                 size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *s = reinterpret_cast<const uint8_t *>(
        reinterpret_cast<const uint8_t *>(src) + y * src_stride);
    uint8_t *d = reinterpret_cast<uint8_t *>(
        reinterpret_cast<uint8_t *>(dst) + y * dst_stride);
    for (size_t x = 0; x < width; ++x) {
      uint8_t r = s[x * 4 + 0], g = s[x * 4 + 1];
      uint8_t b = s[x * 4 + 2], a = s[x * 4 + 3];
      d[x * 4 + 0] = b;
      d[x * 4 + 1] = g;
      d[x * 4 + 2] = r;
      d[x * 4 + 3] = a;
    }
  }
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
rgb_to_bgra_u8(const uint8_t *src, size_t src_stride, uint8_t *dst,
               size_t dst_stride, size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *s = reinterpret_cast<const uint8_t *>(
        reinterpret_cast<const uint8_t *>(src) + y * src_stride);
    uint8_t *d = reinterpret_cast<uint8_t *>(
        reinterpret_cast<uint8_t *>(dst) + y * dst_stride);
    for (size_t x = 0; x < width; ++x) {
      d[x * 4 + 0] = s[x * 3 + 2];
      d[x * 4 + 1] = s[x * 3 + 1];
      d[x * 4 + 2] = s[x * 3 + 0];
      d[x * 4 + 3] = 0xFF;
    }
  }
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t
rgb_to_rgba_u8(const uint8_t *src, size_t src_stride, uint8_t *dst,
               size_t dst_stride, size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *s = reinterpret_cast<const uint8_t *>(
        reinterpret_cast<const uint8_t *>(src) + y * src_stride);
    uint8_t *d = reinterpret_cast<uint8_t *>(
        reinterpret_cast<uint8_t *>(dst) + y * dst_stride);
    for (size_t x = 0; x < width; ++x) {
      std::memcpy(d + x * 4, s + x * 3, 3);
      d[x * 4 + 3] = 0xFF;
    }
  }
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgba_to_bgr_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *s = reinterpret_cast<const uint8_t *>(
        reinterpret_cast<const uint8_t *>(src) + y * src_stride);
    uint8_t *d = reinterpret_cast<uint8_t *>(
        reinterpret_cast<uint8_t *>(dst) + y * dst_stride);
    for (size_t x = 0; x < width; ++x) {
      d[x * 3 + 0] = s[x * 4 + 2];
      d[x * 3 + 1] = s[x * 4 + 1];
      d[x * 3 + 2] = s[x * 4 + 0];
    }
  }
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rgba_to_rgb_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *s = reinterpret_cast<const uint8_t *>(
        reinterpret_cast<const uint8_t *>(src) + y * src_stride);
    uint8_t *d = reinterpret_cast<uint8_t *>(
        reinterpret_cast<uint8_t *>(dst) + y * dst_stride);
    for (size_t x = 0; x < width; ++x) {
      std::memcpy(d + x * 3, s + x * 4, 3);
    }
  }
  return KLEIDICV_OK;
}

#endif  // __riscv_vector

}  // namespace kleidicv::neon
