// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <cmath>
#include <cstring>
#include <limits>

#include "kleidicv/arithmetics/exp_constants.h"
#include "kleidicv/kleidicv.h"

#include "kleidicv/rvv.h"

namespace kleidicv::neon {

#ifdef __riscv_vector

// RVV vectorized exp(x) for float, matching the Neon polynomial algorithm
// from exp_neon.cpp.
//
// Algorithm: exp(x) = 2^n * poly(r)
//   where n = round(x / ln2), r = x - n*ln2 (Cody-Waite reduction)
//   poly(r) is a degree-6 minimax polynomial with max error < 1 ULP.
//
// Special cases (|n| > 126): decompose 2^n into s1*s2 to avoid intermediate
// overflow/underflow, matching the Neon specialcase() logic.

static inline vfloat32m1_t rvv_exp_f32(const float* src, size_t vl) {
  using namespace kleidicv::exp_f32;

  vfloat32m1_t x = __riscv_vle32_v_f32m1(src, vl);

  // z = kShift + x * kInvLn2
  // The magic shift trick: adding kShift (2^23 * 1.5) rounds to nearest
  // integer in the low bits of z's IEEE representation.
  vfloat32m1_t z = __riscv_vfmv_v_f_f32m1(kShift, vl);
  z = __riscv_vfmacc_vf_f32m1(z, kInvLn2, x, vl);

  // n = z - kShift (recover the rounded integer as float)
  vfloat32m1_t n = __riscv_vfsub_vf_f32m1(z, kShift, vl);

  // Cody-Waite reduction: r = x - n*kLn2Hi - n*kLn2Lo
  vfloat32m1_t r = __riscv_vfmacc_vf_f32m1(x, -kLn2Hi, n, vl);
  r = __riscv_vfmacc_vf_f32m1(r, -kLn2Lo, n, vl);

  // Extract exponent bits: e = reinterpret_u32(z) << 23
  vuint32m1_t z_u32 = __riscv_vreinterpret_v_f32m1_u32m1(z);
  vuint32m1_t e = __riscv_vsll_vx_u32m1(z_u32, 23, vl);

  // scale = reinterpret_f32(e + 0x3f800000) = 2^n
  vuint32m1_t scale_u32 =
      __riscv_vadd_vx_u32m1(e, 0x3f800000u, vl);
  vfloat32m1_t scale = __riscv_vreinterpret_v_u32m1_f32m1(scale_u32);

  // Check for special cases: |n| > 126
  vfloat32m1_t abs_n = __riscv_vfabs_v_f32m1(n, vl);
  vbool32_t cmp = __riscv_vmfgt_vf_f32m1_b32(abs_n, 126.0f, vl);

  // Polynomial evaluation (Horner's scheme, identical to Neon):
  //   p = kPoly[0]*r + kPoly[1]
  //   p = p*r + kPoly[2]
  //   p = p*r + kPoly[3]
  //   p = p*r + kPoly[4]
  //   p = p*r + 1.0
  //   p = p*r + 1.0
  vfloat32m1_t poly = __riscv_vfmv_v_f_f32m1(kPoly[1], vl);
  poly = __riscv_vfmacc_vf_f32m1(poly, kPoly[0], r, vl);

  vfloat32m1_t v_c2 = __riscv_vfmv_v_f_f32m1(kPoly[2], vl);
  poly = __riscv_vfmadd_vv_f32m1(poly, r, v_c2, vl);

  vfloat32m1_t v_c3 = __riscv_vfmv_v_f_f32m1(kPoly[3], vl);
  poly = __riscv_vfmadd_vv_f32m1(poly, r, v_c3, vl);

  vfloat32m1_t v_c4 = __riscv_vfmv_v_f_f32m1(kPoly[4], vl);
  poly = __riscv_vfmadd_vv_f32m1(poly, r, v_c4, vl);

  vfloat32m1_t v_one = __riscv_vfmv_v_f_f32m1(1.0f, vl);
  poly = __riscv_vfmadd_vv_f32m1(poly, r, v_one, vl);
  poly = __riscv_vfmadd_vv_f32m1(poly, r, v_one, vl);

  // Check if any element needs special case handling
  long any_special = __riscv_vcpop_m_b32(cmp, vl);

  if (__builtin_expect(any_special != 0, 0)) {
    // Special case: 2^n may overflow, break it up into s1*s2.
    // b = (n <= 0) ? 0x83000000 : 0
    vbool32_t n_le_zero = __riscv_vmfle_vf_f32m1_b32(n, 0.0f, vl);
    vuint32m1_t v_zero_u32 = __riscv_vmv_v_x_u32m1(0u, vl);
    vuint32m1_t v_0x83 = __riscv_vmv_v_x_u32m1(0x83000000u, vl);
    vuint32m1_t b = __riscv_vmerge_vvm_u32m1(v_zero_u32, v_0x83, n_le_zero, vl);

    // s1 = reinterpret_f32(0x7f000000 + b)
    vuint32m1_t s1_u32 = __riscv_vadd_vx_u32m1(b, 0x7f000000u, vl);
    vfloat32m1_t s1 = __riscv_vreinterpret_v_u32m1_f32m1(s1_u32);

    // s2 = reinterpret_f32(e - b)
    vuint32m1_t s2_u32 = __riscv_vsub_vv_u32m1(e, b, vl);
    vfloat32m1_t s2 = __riscv_vreinterpret_v_u32m1_f32m1(s2_u32);

    // r1 = s1 * s1  (overflow/underflow result for |n| > 192)
    vfloat32m1_t r1 = __riscv_vfmul_vv_f32m1(s1, s1, vl);

    // r0 = (poly * s1) * s2  (normal special-case result)
    vfloat32m1_t r0 = __riscv_vfmul_vv_f32m1(poly, s1, vl);
    r0 = __riscv_vfmul_vv_f32m1(r0, s2, vl);

    // Select: if |n| > 192 use r1 (overflow/underflow), else use r0
    vbool32_t cmp2 = __riscv_vmfgt_vf_f32m1_b32(abs_n, 192.0f, vl);
    vuint32m1_t r0_u32 = __riscv_vreinterpret_v_f32m1_u32m1(r0);
    vuint32m1_t r1_u32 = __riscv_vreinterpret_v_f32m1_u32m1(r1);
    vuint32m1_t special_u32 =
        __riscv_vmerge_vvm_u32m1(r0_u32, r1_u32, cmp2, vl);
    vfloat32m1_t special = __riscv_vreinterpret_v_u32m1_f32m1(special_u32);

    // For elements where |n| > 126, use special result;
    // for others, use scale * poly.
    vfloat32m1_t normal_result = __riscv_vfmul_vv_f32m1(scale, poly, vl);
    vuint32m1_t normal_u32 = __riscv_vreinterpret_v_f32m1_u32m1(normal_result);
    vuint32m1_t sp_u32 = __riscv_vreinterpret_v_f32m1_u32m1(special);
    vuint32m1_t result_u32 =
        __riscv_vmerge_vvm_u32m1(normal_u32, sp_u32, cmp, vl);
    return __riscv_vreinterpret_v_u32m1_f32m1(result_u32);
  }

  // Normal path: result = scale * poly
  return __riscv_vfmul_vv_f32m1(scale, poly, vl);
}

#endif  // __riscv_vector

template <typename T>
KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t exp(const T* src, size_t src_stride,
                                              T* dst, size_t dst_stride,
                                              size_t width, size_t height) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);

#ifdef __riscv_vector
  for (size_t row = 0; row < height; ++row) {
    const T* s = row_ptr(src, src_stride, row);
    T* d = row_ptr(dst, dst_stride, row);
    size_t i = 0;
    while (i < width) {
      size_t vl = rvv_setvl<T>(width - i);
      vfloat32m1_t result = rvv_exp_f32(s + i, vl);
      rvv_store(d + i, result, vl);
      i += vl;
    }
  }
#else
  // Scalar fallback using expf() for 1-ULP accuracy.
  for (size_t row = 0; row < height; ++row) {
    const T* s = row_ptr(src, src_stride, row);
    T* d = row_ptr(dst, dst_stride, row);
    for (size_t i = 0; i < width; ++i) {
      d[i] = expf(s[i]);
    }
  }
#endif

  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(type)                             \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t exp<type>(         \
      const type* src, size_t src_stride, type* dst, size_t dst_stride, \
      size_t width, size_t height)

KLEIDICV_INSTANTIATE_TEMPLATE(float);

}  // namespace kleidicv::neon
