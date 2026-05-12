// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstdlib>
#include <cstring>
#include <limits>
#include <vector>

#include "kleidicv/rvv.h"
#include "kleidicv/workspace/border_types.h"
#include "median_blur_border_handling.h"

namespace kleidicv::neon {

// Scalar separable 2D filter implementation.
//
// Applies a separable convolution: first the vertical kernel (kernel_y)
// is applied to produce an intermediate buffer with wider type, then
// the horizontal kernel (kernel_x) is applied to produce the final output.
// Both passes use border handling via get_physical_index.
//
// This is a fixed 5x5 separable filter matching the NEON implementation.

template <typename T>
kleidicv_error_t separable_filter_2d_stripe(
    const T *src, size_t src_stride, T *dst, size_t dst_stride, size_t width,
    size_t height, size_t y_begin, size_t y_end, size_t channels,
    const T *kernel_x, size_t /*kernel_width*/, const T *kernel_y,
    size_t /*kernel_height*/, FixedBorderType fixed_border_type) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);
  CHECK_POINTERS(kernel_x, kernel_y);

  if (channels > KLEIDICV_MAXIMUM_CHANNEL_COUNT) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  constexpr size_t ksize = 5;
  constexpr ptrdiff_t half = 2;

  // Use a wider intermediate type to avoid overflow during convolution.
  // For uint8_t -> uint16_t, for uint16_t -> uint32_t.
  using WiderT =
      typename std::conditional<std::is_same<T, uint8_t>::value, uint16_t,
                                uint32_t>::type;

  Rows<const T> src_rows{src, src_stride, channels};
  Rows<T> dst_rows{dst, dst_stride, channels};

  // Intermediate buffer for vertical pass: one row at a time, width * channels.
  // Use std::malloc (not std::vector/operator new) so that the test's
  // MockMallocToFail infrastructure can intercept allocation failures.
  const size_t row_elems = width * channels;
  WiderT *vbuf = static_cast<WiderT *>(
      std::malloc(row_elems * sizeof(WiderT)));
  if (!vbuf) {
    return KLEIDICV_ERROR_ALLOCATION;
  }

  for (size_t y = y_begin; y < y_end; ++y) {
    // --- Vertical pass: apply kernel_y along the column direction ---
    // For each element in (width * channels), sum kernel_y[k] * src_row[k]

    // Precompute physical source row indices for the 5-tap kernel.
    const T *src_row_ptrs[ksize];
    for (size_t k = 0; k < ksize; ++k) {
      ptrdiff_t sy = get_physical_index(
          static_cast<size_t>(static_cast<ptrdiff_t>(y) +
                              static_cast<ptrdiff_t>(k) - half),
          height, fixed_border_type);
      src_row_ptrs[k] = &src_rows.at(sy, 0)[0];
    }

#ifdef __riscv_vector
    if constexpr (std::is_same<T, uint8_t>::value) {
      // T = uint8_t, WiderT = uint16_t.
      // Accumulate in u32 to handle sum of up to 5 products (each up to 65025).
      // Max sum = 5 * 65025 = 325125, fits in u32.
      // Then saturate-narrow u32 -> u16 for vbuf.
      size_t pos = 0;
      while (pos < row_elems) {
        size_t vl = __riscv_vsetvl_e8m1(row_elems - pos);
        vuint32m4_t acc = __riscv_vmv_v_x_u32m4(0, vl);
        for (size_t k = 0; k < ksize; ++k) {
          vuint8m1_t v8 = __riscv_vle8_v_u8m1(src_row_ptrs[k] + pos, vl);
          vuint16m2_t v16 = __riscv_vzext_vf2_u16m2(v8, vl);
          vuint32m4_t v32 = __riscv_vzext_vf2_u32m4(v16, vl);
          acc = __riscv_vmacc_vx_u32m4(acc, static_cast<uint32_t>(kernel_y[k]),
                                       v32, vl);
        }
        // Saturate-narrow u32 -> u16 and store to vbuf.
        vuint16m2_t result =
            __riscv_vnclipu_wx_u16m2(acc, 0, __RISCV_VXRM_RDN, vl);
        __riscv_vse16_v_u16m2(vbuf + pos, result, vl);
        pos += vl;
      }
    } else if constexpr (std::is_same<T, uint16_t>::value) {
      // T = uint16_t, WiderT = uint32_t.
      // Accumulate in u64 to handle sum of up to 5 products (each up to ~4.29e9).
      // Then saturate-narrow u64 -> u32 for vbuf.
      size_t pos = 0;
      while (pos < row_elems) {
        size_t vl = __riscv_vsetvl_e16m1(row_elems - pos);
        vuint64m4_t acc = __riscv_vmv_v_x_u64m4(0, vl);
        for (size_t k = 0; k < ksize; ++k) {
          vuint16m1_t v16 = __riscv_vle16_v_u16m1(src_row_ptrs[k] + pos, vl);
          vuint32m2_t v32 = __riscv_vzext_vf2_u32m2(v16, vl);
          vuint64m4_t v64 = __riscv_vzext_vf2_u64m4(v32, vl);
          acc = __riscv_vmacc_vx_u64m4(acc, static_cast<uint64_t>(kernel_y[k]),
                                       v64, vl);
        }
        // Saturate-narrow u64 -> u32 and store to vbuf.
        vuint32m2_t result =
            __riscv_vnclipu_wx_u32m2(acc, 0, __RISCV_VXRM_RDN, vl);
        __riscv_vse32_v_u32m2(vbuf + pos, result, vl);
        pos += vl;
      }
    }
#else
    // Scalar fallback for the vertical pass.
    for (size_t i = 0; i < row_elems; ++i) {
      WiderT acc = 0;
      bool overflow = false;
      for (size_t k = 0; k < ksize; ++k) {
        WiderT val = static_cast<WiderT>(src_row_ptrs[k][i]);
        WiderT term = val * static_cast<WiderT>(kernel_y[k]);
        WiderT new_acc = acc + term;
        // Saturating add check.
        if (new_acc < acc) {
          acc = std::numeric_limits<WiderT>::max();
          overflow = true;
          break;
        }
        acc = new_acc;
      }
      if (overflow) {
        acc = std::numeric_limits<WiderT>::max();
      }
      vbuf[i] = acc;
    }
#endif

    // --- Horizontal pass: apply kernel_x along the row direction ---
    // For each (x, ch), sum kernel_x[k] * vbuf(x + k - half)[ch]
    for (size_t x = 0; x < width; ++x) {
      for (size_t ch = 0; ch < channels; ++ch) {
        // Use an even wider accumulator to avoid overflow in the
        // horizontal pass.
        uint64_t acc = 0;
        bool overflow = false;
        for (size_t k = 0; k < ksize; ++k) {
          ptrdiff_t sx = get_physical_index(
              static_cast<size_t>(static_cast<ptrdiff_t>(x) +
                                  static_cast<ptrdiff_t>(k) - half),
              width, fixed_border_type);
          uint64_t val = static_cast<uint64_t>(
              vbuf[static_cast<size_t>(sx) * channels + ch]);
          uint64_t term = val * static_cast<uint64_t>(kernel_x[k]);
          uint64_t new_acc = acc + term;
          if (new_acc < acc) {
            overflow = true;
            break;
          }
          acc = new_acc;
        }

        T result;
        if (overflow || acc > static_cast<uint64_t>(std::numeric_limits<T>::max())) {
          result = std::numeric_limits<T>::max();
        } else {
          result = static_cast<T>(acc);
        }
        dst_rows.at(static_cast<ptrdiff_t>(y),
                    static_cast<ptrdiff_t>(x))[static_cast<ptrdiff_t>(ch)] =
            result;
      }
    }
  }

  std::free(vbuf);
  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                                  \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t                         \
  separable_filter_2d_stripe<type>(                                          \
      const type *src, size_t src_stride, type *dst, size_t dst_stride,      \
      size_t width, size_t height, size_t y_begin, size_t y_end,             \
      size_t channels, const type *kernel_x, size_t, const type *kernel_y,   \
      size_t, FixedBorderType fixed_border_type)

KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint16_t);

}  // namespace kleidicv::neon
