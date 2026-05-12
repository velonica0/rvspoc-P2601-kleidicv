// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/conversions/gray_to_rgb.h"
#include "kleidicv/kleidicv.h"

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

#ifdef __riscv_vector

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t gray_to_rgb_u8(const uint8_t *src, size_t src_stride,
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
      vuint8m1_t gray = rvv_load(src_row + x, vl);

      // Store R, G, B = gray, gray, gray using segment store
      vuint8m1x3_t rgb_tuple = __riscv_vcreate_v_u8m1x3(gray, gray, gray);
      __riscv_vsseg3e8_v_u8m1x3(dst_row + x * 3, rgb_tuple, vl);

      x += vl;
    }
  }

  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t gray_to_rgba_u8(const uint8_t *src, size_t src_stride,
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
      vuint8m1_t gray = rvv_load(src_row + x, vl);
      vuint8m1_t alpha = rvv_splat(uint8_t(0xFF), vl);

      // Store R, G, B, A = gray, gray, gray, 0xFF using segment store
      vuint8m1x4_t rgba_tuple = __riscv_vcreate_v_u8m1x4(gray, gray, gray, alpha);
      __riscv_vsseg4e8_v_u8m1x4(dst_row + x * 4, rgba_tuple, vl);

      x += vl;
    }
  }

  return KLEIDICV_OK;
}

#else  // scalar fallback

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t gray_to_rgb_u8(const uint8_t *src, size_t src_stride,
                                uint8_t *dst, size_t dst_stride, size_t width,
                                size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *src_row = reinterpret_cast<const uint8_t *>(
        reinterpret_cast<const uint8_t *>(src) + y * src_stride);
    uint8_t *dst_row =
        reinterpret_cast<uint8_t *>(reinterpret_cast<uint8_t *>(dst) +
                                    y * dst_stride);

    for (size_t x = 0; x < width; ++x) {
      dst_row[x * 3 + 0] = src_row[x];
      dst_row[x * 3 + 1] = src_row[x];
      dst_row[x * 3 + 2] = src_row[x];
    }
  }

  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t gray_to_rgba_u8(const uint8_t *src, size_t src_stride,
                                 uint8_t *dst, size_t dst_stride, size_t width,
                                 size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  for (size_t y = 0; y < height; ++y) {
    const uint8_t *src_row = reinterpret_cast<const uint8_t *>(
        reinterpret_cast<const uint8_t *>(src) + y * src_stride);
    uint8_t *dst_row =
        reinterpret_cast<uint8_t *>(reinterpret_cast<uint8_t *>(dst) +
                                    y * dst_stride);

    for (size_t x = 0; x < width; ++x) {
      dst_row[x * 4 + 0] = src_row[x];
      dst_row[x * 4 + 1] = src_row[x];
      dst_row[x * 4 + 2] = src_row[x];
      dst_row[x * 4 + 3] = 0xFF;
    }
  }

  return KLEIDICV_OK;
}

#endif  // __riscv_vector

}  // namespace kleidicv::neon
