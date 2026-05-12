// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <climits>
#include <limits>

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

template <typename ScalarType>
kleidicv_error_t min_max_loc(const ScalarType *src, size_t src_stride,
                             size_t width, size_t height, size_t *min_offset,
                             size_t *max_offset) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  if (KLEIDICV_UNLIKELY(width == 0 || height == 0)) {
    return KLEIDICV_ERROR_RANGE;
  }

#ifdef __riscv_vector
  // ---- Phase 1: Find the global min and max values (vectorized) ----
  size_t maxvl = rvv_setvl<ScalarType>(width);

  auto v_min = rvv_splat(std::numeric_limits<ScalarType>::max(), maxvl);
  auto v_max = rvv_splat(std::numeric_limits<ScalarType>::lowest(), maxvl);

  for (size_t r = 0; r < height; ++r) {
    const ScalarType *row = row_ptr(src, src_stride, r);
    size_t i = 0;
    while (i < width) {
      size_t vl = rvv_setvl<ScalarType>(width - i);
      auto v = rvv_load(row + i, vl);
      v_min = rvv_min(v_min, v, vl);
      v_max = rvv_max(v_max, v, vl);
      i += vl;
    }
  }

  // Horizontal reduction to extract scalar min and max.
  ScalarType min_val, max_val;

  if constexpr (std::is_same_v<ScalarType, uint8_t>) {
    auto init_min = rvv_splat(std::numeric_limits<uint8_t>::max(), maxvl);
    auto init_max = rvv_splat(std::numeric_limits<uint8_t>::lowest(), maxvl);
    auto rmin = __riscv_vredminu_vs_u8m1_u8m1(v_min, init_min, maxvl);
    auto rmax = __riscv_vredmaxu_vs_u8m1_u8m1(v_max, init_max, maxvl);
    min_val = __riscv_vmv_x_s_u8m1_u8(rmin);
    max_val = __riscv_vmv_x_s_u8m1_u8(rmax);
  } else {
    // Generic scalar fallback for other types within the RVV path.
    min_val = std::numeric_limits<ScalarType>::max();
    max_val = std::numeric_limits<ScalarType>::lowest();
    for (size_t r = 0; r < height; ++r) {
      const ScalarType *row = row_ptr(src, src_stride, r);
      for (size_t c = 0; c < width; ++c) {
        ScalarType val = row[c];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
      }
    }
  }

  // ---- Phase 2: Find locations of first min and max (vectorized search) ----
  size_t min_row = 0, min_col = 0;
  size_t max_row = 0, max_col = 0;
  bool found_min = false, found_max = false;

  if constexpr (std::is_same_v<ScalarType, uint8_t>) {
    for (size_t r = 0; r < height && !(found_min && found_max); ++r) {
      const ScalarType *row = row_ptr(src, src_stride, r);
      size_t i = 0;
      while (i < width && !(found_min && found_max)) {
        size_t vl = rvv_setvl<ScalarType>(width - i);
        auto v = rvv_load(row + i, vl);

        if (!found_min) {
          vbool8_t mask_min = __riscv_vmseq_vx_u8m1_b8(v, min_val, vl);
          long first_min = __riscv_vfirst_m_b8(mask_min, vl);
          if (first_min >= 0) {
            min_row = r;
            min_col = i + static_cast<size_t>(first_min);
            found_min = true;
          }
        }

        if (!found_max) {
          vbool8_t mask_max = __riscv_vmseq_vx_u8m1_b8(v, max_val, vl);
          long first_max = __riscv_vfirst_m_b8(mask_max, vl);
          if (first_max >= 0) {
            max_row = r;
            max_col = i + static_cast<size_t>(first_max);
            found_max = true;
          }
        }

        i += vl;
      }
    }
  } else {
    // Generic scalar location search for other types.
    for (size_t r = 0; r < height && !(found_min && found_max); ++r) {
      const ScalarType *row = row_ptr(src, src_stride, r);
      for (size_t c = 0; c < width && !(found_min && found_max); ++c) {
        ScalarType val = row[c];
        if (!found_min && val == min_val) {
          min_row = r;
          min_col = c;
          found_min = true;
        }
        if (!found_max && val == max_val) {
          max_row = r;
          max_col = c;
          found_max = true;
        }
      }
    }
  }

  if (min_offset) {
    *min_offset = min_row * src_stride + min_col * sizeof(ScalarType);
  }
  if (max_offset) {
    *max_offset = max_row * src_stride + max_col * sizeof(ScalarType);
  }
#else
  // Scalar fallback
  ScalarType cur_min = std::numeric_limits<ScalarType>::max();
  ScalarType cur_max = std::numeric_limits<ScalarType>::lowest();
  size_t min_row = 0, min_col = 0;
  size_t max_row = 0, max_col = 0;

  for (size_t r = 0; r < height; ++r) {
    const ScalarType *row = row_ptr(src, src_stride, r);
    for (size_t c = 0; c < width; ++c) {
      ScalarType val = row[c];
      if (val < cur_min) {
        cur_min = val;
        min_row = r;
        min_col = c;
      }
      if (val > cur_max) {
        cur_max = val;
        max_row = r;
        max_col = c;
      }
    }
  }

  if (min_offset) {
    *min_offset = min_row * src_stride + min_col * sizeof(ScalarType);
  }
  if (max_offset) {
    *max_offset = max_row * src_stride + max_col * sizeof(ScalarType);
  }
#endif
  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                             \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t min_max_loc<type>( \
      const type *src, size_t src_stride, size_t width, size_t height,  \
      size_t *min_offset, size_t *max_offset)

KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);

}  // namespace kleidicv::neon
