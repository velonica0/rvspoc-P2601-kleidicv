// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/conversions/split.h"
#include "kleidicv/kleidicv.h"

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

#ifdef __riscv_vector

// --- RVV split implementations using vlsegNe intrinsics ---

template <typename T>
static kleidicv_error_t split2_rvv(const T *src, size_t src_stride, T *dst0,
                                   size_t dst0_stride, T *dst1,
                                   size_t dst1_stride, size_t width,
                                   size_t height);

template <typename T>
static kleidicv_error_t split3_rvv(const T *src, size_t src_stride, T *dst0,
                                   size_t dst0_stride, T *dst1,
                                   size_t dst1_stride, T *dst2,
                                   size_t dst2_stride, size_t width,
                                   size_t height);

template <typename T>
static kleidicv_error_t split4_rvv(const T *src, size_t src_stride, T *dst0,
                                   size_t dst0_stride, T *dst1,
                                   size_t dst1_stride, T *dst2,
                                   size_t dst2_stride, T *dst3,
                                   size_t dst3_stride, size_t width,
                                   size_t height);

// --- uint8_t specializations ---

template <>
kleidicv_error_t split2_rvv<uint8_t>(const uint8_t *src, size_t src_stride,
                                     uint8_t *dst0, size_t dst0_stride,
                                     uint8_t *dst1, size_t dst1_stride,
                                     size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint8_t *src_row = row_ptr(src, src_stride, y);
    uint8_t *d0 = row_ptr(dst0, dst0_stride, y);
    uint8_t *d1 = row_ptr(dst1, dst1_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e8(width - x);
      vuint8m1x2_t seg = __riscv_vlseg2e8_v_u8m1x2(src_row + x * 2, vl);
      vuint8m1_t v0 = __riscv_vget_v_u8m1x2_u8m1(seg, 0);
      vuint8m1_t v1 = __riscv_vget_v_u8m1x2_u8m1(seg, 1);
      rvv_store(d0 + x, v0, vl);
      rvv_store(d1 + x, v1, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

template <>
kleidicv_error_t split3_rvv<uint8_t>(const uint8_t *src, size_t src_stride,
                                     uint8_t *dst0, size_t dst0_stride,
                                     uint8_t *dst1, size_t dst1_stride,
                                     uint8_t *dst2, size_t dst2_stride,
                                     size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint8_t *src_row = row_ptr(src, src_stride, y);
    uint8_t *d0 = row_ptr(dst0, dst0_stride, y);
    uint8_t *d1 = row_ptr(dst1, dst1_stride, y);
    uint8_t *d2 = row_ptr(dst2, dst2_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e8(width - x);
      vuint8m1x3_t seg = __riscv_vlseg3e8_v_u8m1x3(src_row + x * 3, vl);
      vuint8m1_t v0 = __riscv_vget_v_u8m1x3_u8m1(seg, 0);
      vuint8m1_t v1 = __riscv_vget_v_u8m1x3_u8m1(seg, 1);
      vuint8m1_t v2 = __riscv_vget_v_u8m1x3_u8m1(seg, 2);
      rvv_store(d0 + x, v0, vl);
      rvv_store(d1 + x, v1, vl);
      rvv_store(d2 + x, v2, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

template <>
kleidicv_error_t split4_rvv<uint8_t>(const uint8_t *src, size_t src_stride,
                                     uint8_t *dst0, size_t dst0_stride,
                                     uint8_t *dst1, size_t dst1_stride,
                                     uint8_t *dst2, size_t dst2_stride,
                                     uint8_t *dst3, size_t dst3_stride,
                                     size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint8_t *src_row = row_ptr(src, src_stride, y);
    uint8_t *d0 = row_ptr(dst0, dst0_stride, y);
    uint8_t *d1 = row_ptr(dst1, dst1_stride, y);
    uint8_t *d2 = row_ptr(dst2, dst2_stride, y);
    uint8_t *d3 = row_ptr(dst3, dst3_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e8(width - x);
      vuint8m1x4_t seg = __riscv_vlseg4e8_v_u8m1x4(src_row + x * 4, vl);
      vuint8m1_t v0 = __riscv_vget_v_u8m1x4_u8m1(seg, 0);
      vuint8m1_t v1 = __riscv_vget_v_u8m1x4_u8m1(seg, 1);
      vuint8m1_t v2 = __riscv_vget_v_u8m1x4_u8m1(seg, 2);
      vuint8m1_t v3 = __riscv_vget_v_u8m1x4_u8m1(seg, 3);
      rvv_store(d0 + x, v0, vl);
      rvv_store(d1 + x, v1, vl);
      rvv_store(d2 + x, v2, vl);
      rvv_store(d3 + x, v3, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

// --- uint16_t specializations ---

template <>
kleidicv_error_t split2_rvv<uint16_t>(const uint16_t *src, size_t src_stride,
                                      uint16_t *dst0, size_t dst0_stride,
                                      uint16_t *dst1, size_t dst1_stride,
                                      size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint16_t *src_row = row_ptr(src, src_stride, y);
    uint16_t *d0 = row_ptr(dst0, dst0_stride, y);
    uint16_t *d1 = row_ptr(dst1, dst1_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e16(width - x);
      vuint16m1x2_t seg = __riscv_vlseg2e16_v_u16m1x2(src_row + x * 2, vl);
      vuint16m1_t v0 = __riscv_vget_v_u16m1x2_u16m1(seg, 0);
      vuint16m1_t v1 = __riscv_vget_v_u16m1x2_u16m1(seg, 1);
      rvv_store(d0 + x, v0, vl);
      rvv_store(d1 + x, v1, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

template <>
kleidicv_error_t split3_rvv<uint16_t>(const uint16_t *src, size_t src_stride,
                                      uint16_t *dst0, size_t dst0_stride,
                                      uint16_t *dst1, size_t dst1_stride,
                                      uint16_t *dst2, size_t dst2_stride,
                                      size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint16_t *src_row = row_ptr(src, src_stride, y);
    uint16_t *d0 = row_ptr(dst0, dst0_stride, y);
    uint16_t *d1 = row_ptr(dst1, dst1_stride, y);
    uint16_t *d2 = row_ptr(dst2, dst2_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e16(width - x);
      vuint16m1x3_t seg = __riscv_vlseg3e16_v_u16m1x3(src_row + x * 3, vl);
      vuint16m1_t v0 = __riscv_vget_v_u16m1x3_u16m1(seg, 0);
      vuint16m1_t v1 = __riscv_vget_v_u16m1x3_u16m1(seg, 1);
      vuint16m1_t v2 = __riscv_vget_v_u16m1x3_u16m1(seg, 2);
      rvv_store(d0 + x, v0, vl);
      rvv_store(d1 + x, v1, vl);
      rvv_store(d2 + x, v2, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

template <>
kleidicv_error_t split4_rvv<uint16_t>(const uint16_t *src, size_t src_stride,
                                      uint16_t *dst0, size_t dst0_stride,
                                      uint16_t *dst1, size_t dst1_stride,
                                      uint16_t *dst2, size_t dst2_stride,
                                      uint16_t *dst3, size_t dst3_stride,
                                      size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint16_t *src_row = row_ptr(src, src_stride, y);
    uint16_t *d0 = row_ptr(dst0, dst0_stride, y);
    uint16_t *d1 = row_ptr(dst1, dst1_stride, y);
    uint16_t *d2 = row_ptr(dst2, dst2_stride, y);
    uint16_t *d3 = row_ptr(dst3, dst3_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e16(width - x);
      vuint16m1x4_t seg = __riscv_vlseg4e16_v_u16m1x4(src_row + x * 4, vl);
      vuint16m1_t v0 = __riscv_vget_v_u16m1x4_u16m1(seg, 0);
      vuint16m1_t v1 = __riscv_vget_v_u16m1x4_u16m1(seg, 1);
      vuint16m1_t v2 = __riscv_vget_v_u16m1x4_u16m1(seg, 2);
      vuint16m1_t v3 = __riscv_vget_v_u16m1x4_u16m1(seg, 3);
      rvv_store(d0 + x, v0, vl);
      rvv_store(d1 + x, v1, vl);
      rvv_store(d2 + x, v2, vl);
      rvv_store(d3 + x, v3, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

// --- uint32_t specializations ---

template <>
kleidicv_error_t split2_rvv<uint32_t>(const uint32_t *src, size_t src_stride,
                                      uint32_t *dst0, size_t dst0_stride,
                                      uint32_t *dst1, size_t dst1_stride,
                                      size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint32_t *src_row = row_ptr(src, src_stride, y);
    uint32_t *d0 = row_ptr(dst0, dst0_stride, y);
    uint32_t *d1 = row_ptr(dst1, dst1_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e32(width - x);
      vuint32m1x2_t seg = __riscv_vlseg2e32_v_u32m1x2(src_row + x * 2, vl);
      vuint32m1_t v0 = __riscv_vget_v_u32m1x2_u32m1(seg, 0);
      vuint32m1_t v1 = __riscv_vget_v_u32m1x2_u32m1(seg, 1);
      rvv_store(d0 + x, v0, vl);
      rvv_store(d1 + x, v1, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

template <>
kleidicv_error_t split3_rvv<uint32_t>(const uint32_t *src, size_t src_stride,
                                      uint32_t *dst0, size_t dst0_stride,
                                      uint32_t *dst1, size_t dst1_stride,
                                      uint32_t *dst2, size_t dst2_stride,
                                      size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint32_t *src_row = row_ptr(src, src_stride, y);
    uint32_t *d0 = row_ptr(dst0, dst0_stride, y);
    uint32_t *d1 = row_ptr(dst1, dst1_stride, y);
    uint32_t *d2 = row_ptr(dst2, dst2_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e32(width - x);
      vuint32m1x3_t seg = __riscv_vlseg3e32_v_u32m1x3(src_row + x * 3, vl);
      vuint32m1_t v0 = __riscv_vget_v_u32m1x3_u32m1(seg, 0);
      vuint32m1_t v1 = __riscv_vget_v_u32m1x3_u32m1(seg, 1);
      vuint32m1_t v2 = __riscv_vget_v_u32m1x3_u32m1(seg, 2);
      rvv_store(d0 + x, v0, vl);
      rvv_store(d1 + x, v1, vl);
      rvv_store(d2 + x, v2, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

template <>
kleidicv_error_t split4_rvv<uint32_t>(const uint32_t *src, size_t src_stride,
                                      uint32_t *dst0, size_t dst0_stride,
                                      uint32_t *dst1, size_t dst1_stride,
                                      uint32_t *dst2, size_t dst2_stride,
                                      uint32_t *dst3, size_t dst3_stride,
                                      size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint32_t *src_row = row_ptr(src, src_stride, y);
    uint32_t *d0 = row_ptr(dst0, dst0_stride, y);
    uint32_t *d1 = row_ptr(dst1, dst1_stride, y);
    uint32_t *d2 = row_ptr(dst2, dst2_stride, y);
    uint32_t *d3 = row_ptr(dst3, dst3_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e32(width - x);
      vuint32m1x4_t seg = __riscv_vlseg4e32_v_u32m1x4(src_row + x * 4, vl);
      vuint32m1_t v0 = __riscv_vget_v_u32m1x4_u32m1(seg, 0);
      vuint32m1_t v1 = __riscv_vget_v_u32m1x4_u32m1(seg, 1);
      vuint32m1_t v2 = __riscv_vget_v_u32m1x4_u32m1(seg, 2);
      vuint32m1_t v3 = __riscv_vget_v_u32m1x4_u32m1(seg, 3);
      rvv_store(d0 + x, v0, vl);
      rvv_store(d1 + x, v1, vl);
      rvv_store(d2 + x, v2, vl);
      rvv_store(d3 + x, v3, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

// --- uint64_t specializations ---

template <>
kleidicv_error_t split2_rvv<uint64_t>(const uint64_t *src, size_t src_stride,
                                      uint64_t *dst0, size_t dst0_stride,
                                      uint64_t *dst1, size_t dst1_stride,
                                      size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint64_t *src_row = row_ptr(src, src_stride, y);
    uint64_t *d0 = row_ptr(dst0, dst0_stride, y);
    uint64_t *d1 = row_ptr(dst1, dst1_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e64(width - x);
      vuint64m1x2_t seg = __riscv_vlseg2e64_v_u64m1x2(src_row + x * 2, vl);
      vuint64m1_t v0 = __riscv_vget_v_u64m1x2_u64m1(seg, 0);
      vuint64m1_t v1 = __riscv_vget_v_u64m1x2_u64m1(seg, 1);
      rvv_store(d0 + x, v0, vl);
      rvv_store(d1 + x, v1, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

template <>
kleidicv_error_t split3_rvv<uint64_t>(const uint64_t *src, size_t src_stride,
                                      uint64_t *dst0, size_t dst0_stride,
                                      uint64_t *dst1, size_t dst1_stride,
                                      uint64_t *dst2, size_t dst2_stride,
                                      size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint64_t *src_row = row_ptr(src, src_stride, y);
    uint64_t *d0 = row_ptr(dst0, dst0_stride, y);
    uint64_t *d1 = row_ptr(dst1, dst1_stride, y);
    uint64_t *d2 = row_ptr(dst2, dst2_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e64(width - x);
      vuint64m1x3_t seg = __riscv_vlseg3e64_v_u64m1x3(src_row + x * 3, vl);
      vuint64m1_t v0 = __riscv_vget_v_u64m1x3_u64m1(seg, 0);
      vuint64m1_t v1 = __riscv_vget_v_u64m1x3_u64m1(seg, 1);
      vuint64m1_t v2 = __riscv_vget_v_u64m1x3_u64m1(seg, 2);
      rvv_store(d0 + x, v0, vl);
      rvv_store(d1 + x, v1, vl);
      rvv_store(d2 + x, v2, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

template <>
kleidicv_error_t split4_rvv<uint64_t>(const uint64_t *src, size_t src_stride,
                                      uint64_t *dst0, size_t dst0_stride,
                                      uint64_t *dst1, size_t dst1_stride,
                                      uint64_t *dst2, size_t dst2_stride,
                                      uint64_t *dst3, size_t dst3_stride,
                                      size_t width, size_t height) {
  for (size_t y = 0; y < height; ++y) {
    const uint64_t *src_row = row_ptr(src, src_stride, y);
    uint64_t *d0 = row_ptr(dst0, dst0_stride, y);
    uint64_t *d1 = row_ptr(dst1, dst1_stride, y);
    uint64_t *d2 = row_ptr(dst2, dst2_stride, y);
    uint64_t *d3 = row_ptr(dst3, dst3_stride, y);
    size_t x = 0;
    while (x < width) {
      size_t vl = rvv_setvl_e64(width - x);
      vuint64m1x4_t seg = __riscv_vlseg4e64_v_u64m1x4(src_row + x * 4, vl);
      vuint64m1_t v0 = __riscv_vget_v_u64m1x4_u64m1(seg, 0);
      vuint64m1_t v1 = __riscv_vget_v_u64m1x4_u64m1(seg, 1);
      vuint64m1_t v2 = __riscv_vget_v_u64m1x4_u64m1(seg, 2);
      vuint64m1_t v3 = __riscv_vget_v_u64m1x4_u64m1(seg, 3);
      rvv_store(d0 + x, v0, vl);
      rvv_store(d1 + x, v1, vl);
      rvv_store(d2 + x, v2, vl);
      rvv_store(d3 + x, v3, vl);
      x += vl;
    }
  }
  return KLEIDICV_OK;
}

// --- Dispatch function matching the neon interface ---

template <typename ScalarType>
static kleidicv_error_t split_typed(const void *src_void, size_t src_stride,
                                    void **dst_data, const size_t *dst_strides,
                                    size_t width, size_t height,
                                    size_t channels) {
  if (channels < 2) return KLEIDICV_ERROR_RANGE;
  if (channels > 4) return KLEIDICV_ERROR_NOT_IMPLEMENTED;

  CHECK_POINTERS(dst_data, dst_strides);
  MAKE_POINTER_CHECK_ALIGNMENT(const ScalarType, src_data, src_void);
  MAKE_POINTER_CHECK_ALIGNMENT(ScalarType, dst0, dst_data[0]);
  MAKE_POINTER_CHECK_ALIGNMENT(ScalarType, dst1, dst_data[1]);
  CHECK_POINTER_AND_STRIDE(src_data, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst0, dst_strides[0], height);
  CHECK_POINTER_AND_STRIDE(dst1, dst_strides[1], height);
  CHECK_IMAGE_SIZE(width, height);

  switch (channels) {
    case 2:
      return split2_rvv<ScalarType>(src_data, src_stride, dst0, dst_strides[0],
                                    dst1, dst_strides[1], width, height);
    case 3: {
      MAKE_POINTER_CHECK_ALIGNMENT(ScalarType, dst2, dst_data[2]);
      CHECK_POINTER_AND_STRIDE(dst2, dst_strides[2], height);
      return split3_rvv<ScalarType>(src_data, src_stride, dst0, dst_strides[0],
                                    dst1, dst_strides[1], dst2, dst_strides[2],
                                    width, height);
    }
    case 4: {
      MAKE_POINTER_CHECK_ALIGNMENT(ScalarType, dst2, dst_data[2]);
      MAKE_POINTER_CHECK_ALIGNMENT(ScalarType, dst3, dst_data[3]);
      CHECK_POINTER_AND_STRIDE(dst2, dst_strides[2], height);
      CHECK_POINTER_AND_STRIDE(dst3, dst_strides[3], height);
      return split4_rvv<ScalarType>(src_data, src_stride, dst0, dst_strides[0],
                                    dst1, dst_strides[1], dst2, dst_strides[2],
                                    dst3, dst_strides[3], width, height);
    }
    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t split(const void *src_data, size_t src_stride, void **dst_data,
                       const size_t *dst_strides, size_t width, size_t height,
                       size_t channels, size_t element_size) {
  switch (element_size) {
    case sizeof(uint8_t):
      return split_typed<uint8_t>(src_data, src_stride, dst_data, dst_strides,
                                  width, height, channels);
    case sizeof(uint16_t):
      return split_typed<uint16_t>(src_data, src_stride, dst_data, dst_strides,
                                   width, height, channels);
    case sizeof(uint32_t):
      return split_typed<uint32_t>(src_data, src_stride, dst_data, dst_strides,
                                   width, height, channels);
    case sizeof(uint64_t):
      return split_typed<uint64_t>(src_data, src_stride, dst_data, dst_strides,
                                   width, height, channels);
    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
}

#else  // scalar fallback

template <typename ScalarType>
static kleidicv_error_t split_scalar(const void *src_void, size_t src_stride,
                                     void **dst_data, const size_t *dst_strides,
                                     size_t width, size_t height,
                                     size_t channels) {
  if (channels < 2) return KLEIDICV_ERROR_RANGE;
  if (channels > 4) return KLEIDICV_ERROR_NOT_IMPLEMENTED;

  CHECK_POINTERS(dst_data, dst_strides);
  MAKE_POINTER_CHECK_ALIGNMENT(const ScalarType, src_data, src_void);
  MAKE_POINTER_CHECK_ALIGNMENT(ScalarType, dst0, dst_data[0]);
  MAKE_POINTER_CHECK_ALIGNMENT(ScalarType, dst1, dst_data[1]);
  CHECK_POINTER_AND_STRIDE(src_data, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst0, dst_strides[0], height);
  CHECK_POINTER_AND_STRIDE(dst1, dst_strides[1], height);
  CHECK_IMAGE_SIZE(width, height);

  ScalarType *dsts[4] = {dst0, dst1, nullptr, nullptr};
  size_t strides[4] = {dst_strides[0], dst_strides[1], 0, 0};

  for (size_t c = 2; c < channels && c < 4; ++c) {
    MAKE_POINTER_CHECK_ALIGNMENT(ScalarType, dc, dst_data[c]);
    CHECK_POINTER_AND_STRIDE(dc, dst_strides[c], height);
    dsts[c] = dc;
    strides[c] = dst_strides[c];
  }

  for (size_t y = 0; y < height; ++y) {
    const ScalarType *s = reinterpret_cast<const ScalarType *>(
        reinterpret_cast<const uint8_t *>(src_data) + y * src_stride);
    for (size_t x = 0; x < width; ++x) {
      for (size_t c = 0; c < channels; ++c) {
        ScalarType *d = reinterpret_cast<ScalarType *>(
            reinterpret_cast<uint8_t *>(dsts[c]) + y * strides[c]);
        d[x] = s[x * channels + c];
      }
    }
  }
  return KLEIDICV_OK;
}

KLEIDICV_TARGET_FN_ATTRS
kleidicv_error_t split(const void *src_data, size_t src_stride, void **dst_data,
                       const size_t *dst_strides, size_t width, size_t height,
                       size_t channels, size_t element_size) {
  switch (element_size) {
    case sizeof(uint8_t):
      return split_scalar<uint8_t>(src_data, src_stride, dst_data, dst_strides,
                                   width, height, channels);
    case sizeof(uint16_t):
      return split_scalar<uint16_t>(src_data, src_stride, dst_data, dst_strides,
                                    width, height, channels);
    case sizeof(uint32_t):
      return split_scalar<uint32_t>(src_data, src_stride, dst_data, dst_strides,
                                    width, height, channels);
    case sizeof(uint64_t):
      return split_scalar<uint64_t>(src_data, src_stride, dst_data, dst_strides,
                                    width, height, channels);
    default:
      return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
}

#endif  // __riscv_vector

}  // namespace kleidicv::neon
