// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <type_traits>

#include "kleidicv/kleidicv.h"

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

#ifdef __riscv_vector
// Signed int32 absdiff: widen to int64 to avoid overflow, then narrow back
// with saturation.
static inline vint32m1_t rvv_absdiff_i32(vint32m1_t a, vint32m1_t b,
                                          size_t vl) {
  // Widen to i64 to avoid overflow
  vint64m2_t wa = __riscv_vsext_vf2_i64m2(a, vl);
  vint64m2_t wb = __riscv_vsext_vf2_i64m2(b, vl);
  vint64m2_t diff = __riscv_vsub_vv_i64m2(wa, wb, vl);
  vint64m2_t neg = __riscv_vneg_v_i64m2(diff, vl);
  vint64m2_t abs_diff = __riscv_vmax_vv_i64m2(diff, neg, vl);
  // Narrow back with saturation
  return __riscv_vnclip_wx_i32m1(abs_diff, 0, __RISCV_VXRM_RDN, vl);
}
#endif

template <typename T>
kleidicv_error_t saturating_absdiff(const T *src_a, size_t src_a_stride,
                                    const T *src_b, size_t src_b_stride, T *dst,
                                    size_t dst_stride, size_t width,
                                    size_t height) {
  CHECK_POINTER_AND_STRIDE(src_a, src_a_stride, height);
  CHECK_POINTER_AND_STRIDE(src_b, src_b_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

#ifdef __riscv_vector
  for (size_t row = 0; row < height; ++row) {
    const T *sa = row_ptr(src_a, src_a_stride, row);
    const T *sb = row_ptr(src_b, src_b_stride, row);
    T *d = row_ptr(dst, dst_stride, row);
    size_t i = 0;
    while (i < width) {
      size_t vl = rvv_setvl<T>(width - i);
      auto va = rvv_load(sa + i, vl);
      auto vb = rvv_load(sb + i, vl);
      if constexpr (std::is_same_v<T, int32_t>) {
        auto vr = rvv_absdiff_i32(va, vb, vl);
        rvv_store(d + i, vr, vl);
      } else {
        auto vr = rvv_absdiff(va, vb, vl);
        rvv_store(d + i, vr, vl);
      }
      i += vl;
    }
  }
#else
  for (size_t row = 0; row < height; ++row) {
    const T *sa = reinterpret_cast<const T *>(
        reinterpret_cast<const uint8_t *>(src_a) + row * src_a_stride);
    const T *sb = reinterpret_cast<const T *>(
        reinterpret_cast<const uint8_t *>(src_b) + row * src_b_stride);
    T *d = reinterpret_cast<T *>(reinterpret_cast<uint8_t *>(dst) +
                                 row * dst_stride);
    for (size_t i = 0; i < width; ++i) {
      using UnsignedT = std::make_unsigned_t<T>;
      UnsignedT u_a = static_cast<UnsignedT>(sa[i]);
      UnsignedT u_b = static_cast<UnsignedT>(sb[i]);
      UnsignedT difference = sa[i] > sb[i] ? u_a - u_b : u_b - u_a;
      d[i] = saturating_cast<UnsignedT, T>(difference);
    }
  }
#endif

  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                                    \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t saturating_absdiff<type>( \
      const type *src_a, size_t src_a_stride, const type *src_b,               \
      size_t src_b_stride, type *dst, size_t dst_stride, size_t width,         \
      size_t height)

KLEIDICV_INSTANTIATE_TEMPLATE(uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(uint16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int16_t);
KLEIDICV_INSTANTIATE_TEMPLATE(int32_t);

}  // namespace kleidicv::neon
