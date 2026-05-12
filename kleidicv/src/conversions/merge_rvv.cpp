// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/conversions/merge.h"
#include "kleidicv/kleidicv.h"

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

#ifdef __riscv_vector

// --- RVV merge implementations using vssegNe intrinsics ---

template <typename T>
static kleidicv_error_t merge2_rvv(const T *src0, size_t src0_stride,
                                   const T *src1, size_t src1_stride, T *dst,
                                   size_t dst_stride, size_t width,
                                   size_t height);

template <typename T>
static kleidicv_error_t merge3_rvv(const T *src0, size_t src0_stride,
                                   const T *src1, size_t src1_stride,
                                   const T *src2, size_t src2_stride, T *dst,
                                   size_t dst_stride, size_t width,
                                   size_t height);

template <typename T>
static kleidicv_error_t merge4_rvv(const T *src0, size_t src0_stride,
                                   const T *src1, size_t src1_stride,
                                   const T *src2, size_t src2_stride,
                                   const T *src3, size_t src3_stride, T *dst,
                                   size_t dst_stride, size_t width,
                                   size_t height);

// --- uint8_t specializations ---

template <>
kleidicv_error_t merge2_rvv<uint8_t>(const uint8_t *src0, size_t src0_stride,
                                     const uint8_t *src1, size_t src1_stride,
                                     uint8_t *dst, size_t dst_stride,
                                     size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint8_t *s0 = row_ptr(src0, src0_stride, y);
    const uint8_t *s1 = row_ptr(src1, src1_stride, y);
    uint8_t *d = row_ptr(dst, dst_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e8(width - x);
      vuint8m1_t v0 = rvv_load(s0 + x, vl);
      vuint8m1_t v1 = rvv_load(s1 + x, vl);
      vuint8m1x2_t out = __riscv_vcreate_v_u8m1x2(v0, v1);
      __riscv_vsseg2e8_v_u8m1x2(d + x * 2, out, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

template <>
kleidicv_error_t merge3_rvv<uint8_t>(const uint8_t *src0, size_t src0_stride,
                                     const uint8_t *src1, size_t src1_stride,
                                     const uint8_t *src2, size_t src2_stride,
                                     uint8_t *dst, size_t dst_stride,
                                     size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint8_t *s0 = row_ptr(src0, src0_stride, y);
    const uint8_t *s1 = row_ptr(src1, src1_stride, y);
    const uint8_t *s2 = row_ptr(src2, src2_stride, y);
    uint8_t *d = row_ptr(dst, dst_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e8(width - x);
      vuint8m1_t v0 = rvv_load(s0 + x, vl);
      vuint8m1_t v1 = rvv_load(s1 + x, vl);
      vuint8m1_t v2 = rvv_load(s2 + x, vl);
      vuint8m1x3_t out = __riscv_vcreate_v_u8m1x3(v0, v1, v2);
      __riscv_vsseg3e8_v_u8m1x3(d + x * 3, out, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

template <>
kleidicv_error_t merge4_rvv<uint8_t>(const uint8_t *src0, size_t src0_stride,
                                     const uint8_t *src1, size_t src1_stride,
                                     const uint8_t *src2, size_t src2_stride,
                                     const uint8_t *src3, size_t src3_stride,
                                     uint8_t *dst, size_t dst_stride,
                                     size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint8_t *s0 = row_ptr(src0, src0_stride, y);
    const uint8_t *s1 = row_ptr(src1, src1_stride, y);
    const uint8_t *s2 = row_ptr(src2, src2_stride, y);
    const uint8_t *s3 = row_ptr(src3, src3_stride, y);
    uint8_t *d = row_ptr(dst, dst_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e8(width - x);
      vuint8m1_t v0 = rvv_load(s0 + x, vl);
      vuint8m1_t v1 = rvv_load(s1 + x, vl);
      vuint8m1_t v2 = rvv_load(s2 + x, vl);
      vuint8m1_t v3 = rvv_load(s3 + x, vl);
      vuint8m1x4_t out = __riscv_vcreate_v_u8m1x4(v0, v1, v2, v3);
      __riscv_vsseg4e8_v_u8m1x4(d + x * 4, out, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

// --- uint16_t specializations ---

template <>
kleidicv_error_t merge2_rvv<uint16_t>(const uint16_t *src0, size_t src0_stride,
                                      const uint16_t *src1, size_t src1_stride,
                                      uint16_t *dst, size_t dst_stride,
                                      size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint16_t *s0 = row_ptr(src0, src0_stride, y);
    const uint16_t *s1 = row_ptr(src1, src1_stride, y);
    uint16_t *d = row_ptr(dst, dst_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e16(width - x);
      vuint16m1_t v0 = rvv_load(s0 + x, vl);
      vuint16m1_t v1 = rvv_load(s1 + x, vl);
      vuint16m1x2_t out = __riscv_vcreate_v_u16m1x2(v0, v1);
      __riscv_vsseg2e16_v_u16m1x2(d + x * 2, out, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

template <>
kleidicv_error_t merge3_rvv<uint16_t>(const uint16_t *src0, size_t src0_stride,
                                      const uint16_t *src1, size_t src1_stride,
                                      const uint16_t *src2, size_t src2_stride,
                                      uint16_t *dst, size_t dst_stride,
                                      size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint16_t *s0 = row_ptr(src0, src0_stride, y);
    const uint16_t *s1 = row_ptr(src1, src1_stride, y);
    const uint16_t *s2 = row_ptr(src2, src2_stride, y);
    uint16_t *d = row_ptr(dst, dst_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e16(width - x);
      vuint16m1_t v0 = rvv_load(s0 + x, vl);
      vuint16m1_t v1 = rvv_load(s1 + x, vl);
      vuint16m1_t v2 = rvv_load(s2 + x, vl);
      vuint16m1x3_t out = __riscv_vcreate_v_u16m1x3(v0, v1, v2);
      __riscv_vsseg3e16_v_u16m1x3(d + x * 3, out, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

template <>
kleidicv_error_t merge4_rvv<uint16_t>(const uint16_t *src0, size_t src0_stride,
                                      const uint16_t *src1, size_t src1_stride,
                                      const uint16_t *src2, size_t src2_stride,
                                      const uint16_t *src3, size_t src3_stride,
                                      uint16_t *dst, size_t dst_stride,
                                      size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint16_t *s0 = row_ptr(src0, src0_stride, y);
    const uint16_t *s1 = row_ptr(src1, src1_stride, y);
    const uint16_t *s2 = row_ptr(src2, src2_stride, y);
    const uint16_t *s3 = row_ptr(src3, src3_stride, y);
    uint16_t *d = row_ptr(dst, dst_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e16(width - x);
      vuint16m1_t v0 = rvv_load(s0 + x, vl);
      vuint16m1_t v1 = rvv_load(s1 + x, vl);
      vuint16m1_t v2 = rvv_load(s2 + x, vl);
      vuint16m1_t v3 = rvv_load(s3 + x, vl);
      vuint16m1x4_t out = __riscv_vcreate_v_u16m1x4(v0, v1, v2, v3);
      __riscv_vsseg4e16_v_u16m1x4(d + x * 4, out, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

// --- uint32_t specializations ---

template <>
kleidicv_error_t merge2_rvv<uint32_t>(const uint32_t *src0, size_t src0_stride,
                                      const uint32_t *src1, size_t src1_stride,
                                      uint32_t *dst, size_t dst_stride,
                                      size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint32_t *s0 = row_ptr(src0, src0_stride, y);
    const uint32_t *s1 = row_ptr(src1, src1_stride, y);
    uint32_t *d = row_ptr(dst, dst_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e32(width - x);
      vuint32m1_t v0 = rvv_load(s0 + x, vl);
      vuint32m1_t v1 = rvv_load(s1 + x, vl);
      vuint32m1x2_t out = __riscv_vcreate_v_u32m1x2(v0, v1);
      __riscv_vsseg2e32_v_u32m1x2(d + x * 2, out, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

template <>
kleidicv_error_t merge3_rvv<uint32_t>(const uint32_t *src0, size_t src0_stride,
                                      const uint32_t *src1, size_t src1_stride,
                                      const uint32_t *src2, size_t src2_stride,
                                      uint32_t *dst, size_t dst_stride,
                                      size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint32_t *s0 = row_ptr(src0, src0_stride, y);
    const uint32_t *s1 = row_ptr(src1, src1_stride, y);
    const uint32_t *s2 = row_ptr(src2, src2_stride, y);
    uint32_t *d = row_ptr(dst, dst_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e32(width - x);
      vuint32m1_t v0 = rvv_load(s0 + x, vl);
      vuint32m1_t v1 = rvv_load(s1 + x, vl);
      vuint32m1_t v2 = rvv_load(s2 + x, vl);
      vuint32m1x3_t out = __riscv_vcreate_v_u32m1x3(v0, v1, v2);
      __riscv_vsseg3e32_v_u32m1x3(d + x * 3, out, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

template <>
kleidicv_error_t merge4_rvv<uint32_t>(const uint32_t *src0, size_t src0_stride,
                                      const uint32_t *src1, size_t src1_stride,
                                      const uint32_t *src2, size_t src2_stride,
                                      const uint32_t *src3, size_t src3_stride,
                                      uint32_t *dst, size_t dst_stride,
                                      size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint32_t *s0 = row_ptr(src0, src0_stride, y);
    const uint32_t *s1 = row_ptr(src1, src1_stride, y);
    const uint32_t *s2 = row_ptr(src2, src2_stride, y);
    const uint32_t *s3 = row_ptr(src3, src3_stride, y);
    uint32_t *d = row_ptr(dst, dst_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e32(width - x);
      vuint32m1_t v0 = rvv_load(s0 + x, vl);
      vuint32m1_t v1 = rvv_load(s1 + x, vl);
      vuint32m1_t v2 = rvv_load(s2 + x, vl);
      vuint32m1_t v3 = rvv_load(s3 + x, vl);
      vuint32m1x4_t out = __riscv_vcreate_v_u32m1x4(v0, v1, v2, v3);
      __riscv_vsseg4e32_v_u32m1x4(d + x * 4, out, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

// --- uint64_t specializations ---

template <>
kleidicv_error_t merge2_rvv<uint64_t>(const uint64_t *src0, size_t src0_stride,
                                      const uint64_t *src1, size_t src1_stride,
                                      uint64_t *dst, size_t dst_stride,
                                      size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint64_t *s0 = row_ptr(src0, src0_stride, y);
    const uint64_t *s1 = row_ptr(src1, src1_stride, y);
    uint64_t *d = row_ptr(dst, dst_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e64(width - x);
      vuint64m1_t v0 = rvv_load(s0 + x, vl);
      vuint64m1_t v1 = rvv_load(s1 + x, vl);
      vuint64m1x2_t out = __riscv_vcreate_v_u64m1x2(v0, v1);
      __riscv_vsseg2e64_v_u64m1x2(d + x * 2, out, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

template <>
kleidicv_error_t merge3_rvv<uint64_t>(const uint64_t *src0, size_t src0_stride,
                                      const uint64_t *src1, size_t src1_stride,
                                      const uint64_t *src2, size_t src2_stride,
                                      uint64_t *dst, size_t dst_stride,
                                      size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint64_t *s0 = row_ptr(src0, src0_stride, y);
    const uint64_t *s1 = row_ptr(src1, src1_stride, y);
    const uint64_t *s2 = row_ptr(src2, src2_stride, y);
    uint64_t *d = row_ptr(dst, dst_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e64(width - x);
      vuint64m1_t v0 = rvv_load(s0 + x, vl);
      vuint64m1_t v1 = rvv_load(s1 + x, vl);
      vuint64m1_t v2 = rvv_load(s2 + x, vl);
      vuint64m1x3_t out = __riscv_vcreate_v_u64m1x3(v0, v1, v2);
      __riscv_vsseg3e64_v_u64m1x3(d + x * 3, out, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

template <>
kleidicv_error_t merge4_rvv<uint64_t>(const uint64_t *src0, size_t src0_stride,
                                      const uint64_t *src1, size_t src1_stride,
                                      const uint64_t *src2, size_t src2_stride,
                                      const uint64_t *src3, size_t src3_stride,
                                      uint64_t *dst, size_t dst_stride,
                                      size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint64_t *s0 = row_ptr(src0, src0_stride, y);
    const uint64_t *s1 = row_ptr(src1, src1_stride, y);
    const uint64_t *s2 = row_ptr(src2, src2_stride, y);
    const uint64_t *s3 = row_ptr(src3, src3_stride, y);
    uint64_t *d = row_ptr(dst, dst_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e64(width - x);
      vuint64m1_t v0 = rvv_load(s0 + x, vl);
      vuint64m1_t v1 = rvv_load(s1 + x, vl);
      vuint64m1_t v2 = rvv_load(s2 + x, vl);
      vuint64m1_t v3 = rvv_load(s3 + x, vl);
      vuint64m1x4_t out = __riscv_vcreate_v_u64m1x4(v0, v1, v2, v3);
      __riscv_vsseg4e64_v_u64m1x4(d + x * 4, out, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

// --- Dispatch function matching the neon interface ---

template <typename ScalarType>
static kleidicv_error_t merge_typed(const void **srcs, const size_t *src_strides,
                                    void *dst_void, size_t dst_stride,
                                    size_t width, size_t height,
                                    size_t channels) {
  if (channels < 2) return KLEIDICV_ERROR_RANGE;
  if (channels > 4) return KLEIDICV_ERROR_NOT_IMPLEMENTED;

  CHECK_POINTERS(srcs, src_strides);
  MAKE_POINTER_CHECK_ALIGNMENT(const ScalarType, src0, srcs[0]);
  MAKE_POINTER_CHECK_ALIGNMENT(const ScalarType, src1, srcs[1]);
  MAKE_POINTER_CHECK_ALIGNMENT(ScalarType, dst, dst_void);
  CHECK_POINTER_AND_STRIDE(src0, src_strides[0], height);
  CHECK_POINTER_AND_STRIDE(src1, src_strides[1], height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  switch (channels) {
    case 2:
      return merge2_rvv<ScalarType>(src0, src_strides[0], src1, src_strides[1],
                                    dst, dst_stride, width, height);
    case 3: {
      MAKE_POINTER_CHECK_ALIGNMENT(const ScalarType, src2, srcs[2]);
      CHECK_POINTER_AND_STRIDE(src2, src_strides[2], height);
      return merge3_rvv<ScalarType>(src0, src_strides[0], src1, src_strides[1],
                                    src2, src_strides[2], dst, dst_stride,
                                    width, height);
    }
    case 4: {
      MAKE_POINTER_CHECK_ALIGNMENT(const ScalarType, src2, srcs[2]);
      MAKE_POINTER_CHECK_ALIGNMENT(const ScalarType, src3, srcs[3]);
      CHECK_POINTER_AND_STRIDE(src2, src_strides[2], height);
      CHECK_POINTER_AND_STRIDE(src3, src_strides[3], height);
      return merge4_rvv<ScalarType>(src0, src_strides[0], src1, src_strides[1],
                                    src2, src_strides[2], src3, src_strides[3],
                                    dst, dst_stride, width, height);
    }
    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t merge(const void **srcs, const size_t *src_strides, void *dst,
                       size_t dst_stride, size_t width, size_t height,
                       size_t channels, size_t element_size) {
  switch (element_size) {
    case sizeof(uint8_t):
      return merge_typed<uint8_t>(srcs, src_strides, dst, dst_stride, width,
                                  height, channels);
    case sizeof(uint16_t):
      return merge_typed<uint16_t>(srcs, src_strides, dst, dst_stride, width,
                                   height, channels);
    case sizeof(uint32_t):
      return merge_typed<uint32_t>(srcs, src_strides, dst, dst_stride, width,
                                   height, channels);
    case sizeof(uint64_t):
      return merge_typed<uint64_t>(srcs, src_strides, dst, dst_stride, width,
                                   height, channels);
    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
}

#else  // scalar fallback

template <typename ScalarType>
static kleidicv_error_t merge_scalar(const void **srcs,
                                     const size_t *src_strides, void *dst_void,
                                     size_t dst_stride, size_t width,
                                     size_t height, size_t channels) {
  if (channels < 2) return KLEIDICV_ERROR_RANGE;
  if (channels > 4) return KLEIDICV_ERROR_NOT_IMPLEMENTED;

  CHECK_POINTERS(srcs, src_strides);
  MAKE_POINTER_CHECK_ALIGNMENT(const ScalarType, src0, srcs[0]);
  MAKE_POINTER_CHECK_ALIGNMENT(const ScalarType, src1, srcs[1]);
  MAKE_POINTER_CHECK_ALIGNMENT(ScalarType, dst, dst_void);
  CHECK_POINTER_AND_STRIDE(src0, src_strides[0], height);
  CHECK_POINTER_AND_STRIDE(src1, src_strides[1], height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  const ScalarType *sources[4] = {src0, src1, nullptr, nullptr};
  size_t strides[4] = {src_strides[0], src_strides[1], 0, 0};

  for (size_t c = 2; c < channels && c < 4; ++c) {
    MAKE_POINTER_CHECK_ALIGNMENT(const ScalarType, sc, srcs[c]);
    CHECK_POINTER_AND_STRIDE(sc, src_strides[c], height);
    sources[c] = sc;
    strides[c] = src_strides[c];
  }

  for (size_t y = 0; y < height; ++y) {
    ScalarType *d = reinterpret_cast<ScalarType *>(
        reinterpret_cast<uint8_t *>(dst) + y * dst_stride);
    for (size_t x = 0; x < width; ++x) {
      for (size_t c = 0; c < channels; ++c) {
        const ScalarType *s = reinterpret_cast<const ScalarType *>(
            reinterpret_cast<const uint8_t *>(sources[c]) + y * strides[c]);
        d[x * channels + c] = s[x];
      }
    }
  }
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t merge(const void **srcs, const size_t *src_strides, void *dst,
                       size_t dst_stride, size_t width, size_t height,
                       size_t channels, size_t element_size) {
  switch (element_size) {
    case sizeof(uint8_t):
      return merge_scalar<uint8_t>(srcs, src_strides, dst, dst_stride, width,
                                   height, channels);
    case sizeof(uint16_t):
      return merge_scalar<uint16_t>(srcs, src_strides, dst, dst_stride, width,
                                    height, channels);
    case sizeof(uint32_t):
      return merge_scalar<uint32_t>(srcs, src_strides, dst, dst_stride, width,
                                    height, channels);
    case sizeof(uint64_t):
      return merge_scalar<uint64_t>(srcs, src_strides, dst, dst_stride, width,
                                    height, channels);
    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
}

#endif  // __riscv_vector

}  // namespace kleidicv::neon
