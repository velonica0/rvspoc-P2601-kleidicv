// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cassert>

#include "kleidicv/rvv.h"
#include "kleidicv/transform/rotate.h"

namespace kleidicv::neon {

// Scalar rotation using coordinate mapping.
template <typename T, bool k3Channels = false>
static kleidicv_error_t rotate(const void *src_void, size_t src_stride,
                               size_t src_width, size_t src_height,
                               void *dst_void, size_t dst_stride, int angle) {
  MAKE_POINTER_CHECK_ALIGNMENT(const T, src, src_void);
  MAKE_POINTER_CHECK_ALIGNMENT(T, dst, dst_void);
  CHECK_POINTER_AND_STRIDE(src, src_stride, src_height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, src_width);
  CHECK_IMAGE_SIZE(src_width, src_height);

  constexpr size_t ch = k3Channels ? 3 : 1;

  if (angle == 90) {
#ifdef __riscv_vector
    // RVV vectorised 90 CW rotation for single-channel u8/u16/u32.
    // dst[j][height-1-i] = src[i][j]
    // Load vl consecutive elements from src row i starting at column j,
    // then strided-store them to dst rows j..j+vl-1, column height-1-i.
    if constexpr (!k3Channels && sizeof(T) <= 4) {
      for (size_t i = 0; i < src_height; ++i) {
        const T *src_row = row_ptr(src, src_stride, i);
        size_t j = 0;
        while (j < src_width) {
          size_t vl = rvv_setvl<T>(src_width - j);
          // Base: dst row j, column (height-1-i)
          T *dst_col = row_ptr(dst, dst_stride, j) + (src_height - 1 - i);
          if constexpr (sizeof(T) == 1) {
            vuint8m1_t v = __riscv_vle8_v_u8m1(
                reinterpret_cast<const uint8_t *>(src_row + j), vl);
            __riscv_vsse8_v_u8m1(reinterpret_cast<uint8_t *>(dst_col),
                                 static_cast<ptrdiff_t>(dst_stride), v, vl);
          } else if constexpr (sizeof(T) == 2) {
            vuint16m1_t v = __riscv_vle16_v_u16m1(
                reinterpret_cast<const uint16_t *>(src_row + j), vl);
            __riscv_vsse16_v_u16m1(reinterpret_cast<uint16_t *>(dst_col),
                                   static_cast<ptrdiff_t>(dst_stride), v, vl);
          } else {  // sizeof(T) == 4
            vuint32m1_t v = __riscv_vle32_v_u32m1(
                reinterpret_cast<const uint32_t *>(src_row + j), vl);
            __riscv_vsse32_v_u32m1(reinterpret_cast<uint32_t *>(dst_col),
                                   static_cast<ptrdiff_t>(dst_stride), v, vl);
          }
          j += vl;
        }
      }
    } else
#endif  // __riscv_vector
    {
      // Scalar fallback: 90 CW
      for (size_t i = 0; i < src_height; ++i) {
        const T *src_row = row_ptr(src, src_stride, i);
        for (size_t j = 0; j < src_width; ++j) {
          T *dst_pixel =
              row_ptr(dst, dst_stride, j) + (src_height - 1 - i) * ch;
          const T *src_pixel = src_row + j * ch;
          for (size_t c = 0; c < ch; ++c) {
            dst_pixel[c] = src_pixel[c];
          }
        }
      }
    }
  } else {
    // 270 CW (= 90 CCW): dst[width-1-j][i] = src[i][j]
    // Kept scalar — destination rows go in reverse order, making
    // strided stores impractical without a negative stride.
    for (size_t i = 0; i < src_height; ++i) {
      const T *src_row = row_ptr(src, src_stride, i);
      for (size_t j = 0; j < src_width; ++j) {
        T *dst_pixel =
            row_ptr(dst, dst_stride, src_width - 1 - j) + i * ch;
        const T *src_pixel = src_row + j * ch;
        for (size_t c = 0; c < ch; ++c) {
          dst_pixel[c] = src_pixel[c];
        }
      }
    }
  }
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t rotate(const void *src, size_t src_stride, size_t src_width,
                        size_t src_height, void *dst, size_t dst_stride,
                        int angle, size_t pixel_size) {
  if (!rotate_is_implemented(src, dst, angle, pixel_size)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  switch (pixel_size) {
    case sizeof(uint8_t):
      return rotate<uint8_t>(src, src_stride, src_width, src_height, dst,
                             dst_stride, angle);
    case sizeof(uint16_t):
      return rotate<uint16_t>(src, src_stride, src_width, src_height, dst,
                              dst_stride, angle);
    case sizeof(uint32_t):
      return rotate<uint32_t>(src, src_stride, src_width, src_height, dst,
                              dst_stride, angle);
    case sizeof(uint64_t):
      return rotate<uint64_t>(src, src_stride, src_width, src_height, dst,
                              dst_stride, angle);
    case sizeof(uint8_t) * 3:
      return rotate<uint8_t, true>(src, src_stride, src_width, src_height, dst,
                                   dst_stride, angle);
    case sizeof(uint16_t) * 3:
      return rotate<uint16_t, true>(src, src_stride, src_width, src_height, dst,
                                    dst_stride, angle);
    default:
      assert(!"pixel size not implemented");
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
}

}  // namespace kleidicv::neon
