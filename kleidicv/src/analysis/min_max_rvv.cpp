// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <limits>

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

template <typename ScalarType>
kleidicv_error_t min_max(const ScalarType *src, size_t src_stride, size_t width,
                         size_t height, ScalarType *min_value,
                         ScalarType *max_value) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_IMAGE_SIZE(width, height);

  if (KLEIDICV_UNLIKELY(width == 0 || height == 0)) {
    return KLEIDICV_ERROR_RANGE;
  }

#ifdef __riscv_vector
  // Determine the maximum vector length for this element type.
  size_t maxvl = rvv_setvl<ScalarType>(width);

  // Initialize accumulator vectors with identity values for min/max.
  auto v_min = [&]() {
    if constexpr (std::is_same_v<ScalarType, float>)
      return rvv_splat_f32(std::numeric_limits<ScalarType>::max(), maxvl);
    else
      return rvv_splat(std::numeric_limits<ScalarType>::max(), maxvl);
  }();
  auto v_max = [&]() {
    if constexpr (std::is_same_v<ScalarType, float>)
      return rvv_splat_f32(std::numeric_limits<ScalarType>::lowest(), maxvl);
    else
      return rvv_splat(std::numeric_limits<ScalarType>::lowest(), maxvl);
  }();

  // Process each row.
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

  // Horizontal reduction to extract scalar min and max from the vectors.
  // Use the full maxvl so all accumulated lanes participate in the reduction.
  ScalarType cur_min, cur_max;

  if constexpr (std::is_same_v<ScalarType, int8_t>) {
    // For signed reduction, initialize the scalar accumulator to max (for min)
    // and lowest (for max) so the reduction result is purely from v_min/v_max.
    auto init_min = rvv_splat(std::numeric_limits<int8_t>::max(), maxvl);
    auto init_max = rvv_splat(std::numeric_limits<int8_t>::lowest(), maxvl);
    auto rmin = __riscv_vredmin_vs_i8m1_i8m1(v_min, init_min, maxvl);
    auto rmax = __riscv_vredmax_vs_i8m1_i8m1(v_max, init_max, maxvl);
    cur_min = __riscv_vmv_x_s_i8m1_i8(rmin);
    cur_max = __riscv_vmv_x_s_i8m1_i8(rmax);
  } else if constexpr (std::is_same_v<ScalarType, uint8_t>) {
    auto init_min = rvv_splat(std::numeric_limits<uint8_t>::max(), maxvl);
    auto init_max = rvv_splat(std::numeric_limits<uint8_t>::lowest(), maxvl);
    auto rmin = __riscv_vredminu_vs_u8m1_u8m1(v_min, init_min, maxvl);
    auto rmax = __riscv_vredmaxu_vs_u8m1_u8m1(v_max, init_max, maxvl);
    cur_min = __riscv_vmv_x_s_u8m1_u8(rmin);
    cur_max = __riscv_vmv_x_s_u8m1_u8(rmax);
  } else if constexpr (std::is_same_v<ScalarType, int16_t>) {
    auto init_min = rvv_splat(std::numeric_limits<int16_t>::max(), maxvl);
    auto init_max = rvv_splat(std::numeric_limits<int16_t>::lowest(), maxvl);
    auto rmin = __riscv_vredmin_vs_i16m1_i16m1(v_min, init_min, maxvl);
    auto rmax = __riscv_vredmax_vs_i16m1_i16m1(v_max, init_max, maxvl);
    cur_min = __riscv_vmv_x_s_i16m1_i16(rmin);
    cur_max = __riscv_vmv_x_s_i16m1_i16(rmax);
  } else if constexpr (std::is_same_v<ScalarType, uint16_t>) {
    auto init_min = rvv_splat(std::numeric_limits<uint16_t>::max(), maxvl);
    auto init_max = rvv_splat(std::numeric_limits<uint16_t>::lowest(), maxvl);
    auto rmin = __riscv_vredminu_vs_u16m1_u16m1(v_min, init_min, maxvl);
    auto rmax = __riscv_vredmaxu_vs_u16m1_u16m1(v_max, init_max, maxvl);
    cur_min = __riscv_vmv_x_s_u16m1_u16(rmin);
    cur_max = __riscv_vmv_x_s_u16m1_u16(rmax);
  } else if constexpr (std::is_same_v<ScalarType, int32_t>) {
    auto init_min = rvv_splat(std::numeric_limits<int32_t>::max(), maxvl);
    auto init_max = rvv_splat(std::numeric_limits<int32_t>::lowest(), maxvl);
    auto rmin = __riscv_vredmin_vs_i32m1_i32m1(v_min, init_min, maxvl);
    auto rmax = __riscv_vredmax_vs_i32m1_i32m1(v_max, init_max, maxvl);
    cur_min = __riscv_vmv_x_s_i32m1_i32(rmin);
    cur_max = __riscv_vmv_x_s_i32m1_i32(rmax);
  } else if constexpr (std::is_same_v<ScalarType, float>) {
    auto init_min = rvv_splat_f32(std::numeric_limits<float>::max(), maxvl);
    auto init_max = rvv_splat_f32(std::numeric_limits<float>::lowest(), maxvl);
    auto rmin = __riscv_vfredmin_vs_f32m1_f32m1(v_min, init_min, maxvl);
    auto rmax = __riscv_vfredmax_vs_f32m1_f32m1(v_max, init_max, maxvl);
    cur_min = __riscv_vfmv_f_s_f32m1_f32(rmin);
    cur_max = __riscv_vfmv_f_s_f32m1_f32(rmax);
  }

#else
  // Scalar fallback (works for all types, correct on all VLEN)
  ScalarType cur_min = std::numeric_limits<ScalarType>::max();
  ScalarType cur_max = std::numeric_limits<ScalarType>::lowest();

  for (size_t r = 0; r < height; ++r) {
    const ScalarType *row = row_ptr(src, src_stride, r);
    for (size_t c = 0; c < width; ++c) {
      ScalarType val = row[c];
      if (val < cur_min) cur_min = val;
      if (val > cur_max) cur_max = val;
    }
  }
#endif

  if (min_value) {
    *min_value = cur_min;
  }
  if (max_value) {
    *max_value = cur_max;
  }
  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                            \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t min_max<type>(    \
      const type *src, size_t src_stride, size_t width, size_t height, \
      type *min_value, type *max_value)

KLEIDICV_INSTANTIATE_TEMPLATE(int8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int32_t);
KLEIDICV_INSTANTIATE_TEMPLATE(float);

}  // namespace kleidicv::neon
