// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
// SPDX-License-Identifier: Apache-2.0
//
// Benchmark all KleidiCV operators.
// Build against RVV or scalar library to compare speedup.
//
// Usage:
//   c++ -O2 -std=c++17 -I../kleidicv/include -I../build/kleidicv/include \
//       -o bench_rvv bench.cpp -L../build/kleidicv -lkleidicv
//   ./bench_rvv

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>

#include "kleidicv/kleidicv.h"

static double now_ms() {
  return std::chrono::duration<double, std::milli>(
             std::chrono::high_resolution_clock::now().time_since_epoch())
      .count();
}

#define BENCH(label, iters, setup, call)       \
  do {                                         \
    setup;                                     \
    call;                                      \
    double _t0 = now_ms();                     \
    for (int _i = 0; _i < (iters); ++_i) {    \
      call;                                    \
    }                                          \
    double _ms = (now_ms() - _t0) / (iters);  \
    printf("%-35s %8.3f ms\n", label, _ms);    \
  } while (0)

int main() {
  const size_t W = 1920, H = 1080, N = W * H;
  const int IT = 100;

  uint8_t *u8a = (uint8_t *)malloc(N);
  uint8_t *u8b = (uint8_t *)malloc(N);
  uint8_t *u8c = (uint8_t *)malloc(N);
  int8_t *s8a = (int8_t *)malloc(N);
  int8_t *s8b = (int8_t *)malloc(N);
  int8_t *s8c = (int8_t *)malloc(N);
  uint16_t *u16a = (uint16_t *)malloc(N * 2);
  uint16_t *u16b = (uint16_t *)malloc(N * 2);
  uint16_t *u16c = (uint16_t *)malloc(N * 2);
  int16_t *s16a = (int16_t *)malloc(N * 2);
  int16_t *s16b = (int16_t *)malloc(N * 2);
  int16_t *s16c = (int16_t *)malloc(N * 2);
  int32_t *s32a = (int32_t *)malloc(N * 4);
  int32_t *s32b = (int32_t *)malloc(N * 4);
  int32_t *s32c = (int32_t *)malloc(N * 4);
  uint32_t *u32a = (uint32_t *)malloc(N * 4);
  uint32_t *u32b = (uint32_t *)malloc(N * 4);
  uint32_t *u32c = (uint32_t *)malloc(N * 4);
  int64_t *s64a = (int64_t *)malloc(N * 8);
  int64_t *s64b = (int64_t *)malloc(N * 8);
  int64_t *s64c = (int64_t *)malloc(N * 8);
  float *fa = (float *)malloc(N * 4);
  float *fb = (float *)malloc(N * 4);
  float *fc = (float *)malloc(N * 4);

  for (size_t i = 0; i < N; i++) {
    u8a[i] = i & 0xFF;
    u8b[i] = (i * 7) & 0xFF;
    s8a[i] = i & 0x7F;
    s8b[i] = (i * 3) & 0x7F;
    u16a[i] = i & 0xFFFF;
    u16b[i] = (i * 7) & 0xFFFF;
    s16a[i] = i & 0x7FFF;
    s16b[i] = (i * 3) & 0x7FFF;
    s32a[i] = (int32_t)i;
    s32b[i] = (int32_t)(i * 3);
    u32a[i] = (uint32_t)i;
    u32b[i] = (uint32_t)(i * 7);
    s64a[i] = (int64_t)i;
    s64b[i] = (int64_t)(i * 3);
    fa[i] = (float)i / (float)N * 10.0f - 5.0f;
  }

  printf("KleidiCV Benchmark: %zux%zu, %d iterations\n\n", W, H, IT);

  // --- Arithmetic ---
  printf("=== Arithmetic ===\n");
  BENCH("saturating_add_u8", IT, , kleidicv_saturating_add_u8(u8a, W, u8b, W, u8c, W, W, H));
  BENCH("saturating_add_s8", IT, , kleidicv_saturating_add_s8(s8a, W, s8b, W, s8c, W, W, H));
  BENCH("saturating_add_u16", IT, , kleidicv_saturating_add_u16(u16a, W*2, u16b, W*2, u16c, W*2, W, H));
  BENCH("saturating_add_s16", IT, , kleidicv_saturating_add_s16(s16a, W*2, s16b, W*2, s16c, W*2, W, H));
  BENCH("saturating_add_u32", IT, , kleidicv_saturating_add_u32(u32a, W*4, u32b, W*4, u32c, W*4, W, H));
  BENCH("saturating_add_s32", IT, , kleidicv_saturating_add_s32(s32a, W*4, s32b, W*4, s32c, W*4, W, H));
  BENCH("saturating_add_s64", IT, , kleidicv_saturating_add_s64(s64a, W*8, s64b, W*8, s64c, W*8, W, H));
  BENCH("saturating_sub_u8", IT, , kleidicv_saturating_sub_u8(u8a, W, u8b, W, u8c, W, W, H));
  BENCH("saturating_absdiff_u8", IT, , kleidicv_saturating_absdiff_u8(u8a, W, u8b, W, u8c, W, W, H));
  BENCH("saturating_absdiff_s8", IT, , kleidicv_saturating_absdiff_s8(s8a, W, s8b, W, s8c, W, W, H));
  BENCH("saturating_absdiff_s32", IT, , kleidicv_saturating_absdiff_s32(s32a, W*4, s32b, W*4, s32c, W*4, W, H));
  BENCH("bitwise_and_u8", IT, , kleidicv_bitwise_and(u8a, W, u8b, W, u8c, W, W, H));
  BENCH("compare_equal_u8", IT, , kleidicv_compare_equal_u8(u8a, W, u8b, W, u8c, W, W, H));
  BENCH("compare_greater_u8", IT, , kleidicv_compare_greater_u8(u8a, W, u8b, W, u8c, W, W, H));
  BENCH("threshold_binary_u8", IT, , kleidicv_threshold_binary_u8(u8a, W, u8c, W, W, H, 128, 255));
  BENCH("in_range_u8", IT, , kleidicv_in_range_u8(u8a, W, u8c, W, W, H, 50, 200));
  BENCH("scale_u8", IT, , kleidicv_scale_u8(u8a, W, u8c, W, W, H, 0.5, 0.0));
  BENCH("multiply_u8", IT, , kleidicv_saturating_multiply_u8(u8a, W, u8b, W, u8c, W, W, H, 1.0));
  BENCH("multiply_s16", IT, , kleidicv_saturating_multiply_s16(s16a, W*2, s16b, W*2, s16c, W*2, W, H, 1.0));
  BENCH("add_abs_thresh_s16", IT, , kleidicv_saturating_add_abs_with_threshold_s16(s16a, W*2, s16b, W*2, s16c, W*2, W, H, 100));
  BENCH("exp_f32", IT, , kleidicv_exp_f32(fa, W*4, fc, W*4, W, H));

  // --- Conversions ---
  printf("\n=== Conversions ===\n");
  uint8_t *rgb = (uint8_t *)malloc(N * 3);
  uint8_t *bgr = (uint8_t *)malloc(N * 3);
  uint8_t *rgba = (uint8_t *)malloc(N * 4);
  memset(rgb, 128, N * 3);
  BENCH("gray_to_rgb_u8", IT, , kleidicv_gray_to_rgb_u8(u8a, W, rgb, W*3, W, H));
  BENCH("gray_to_rgba_u8", IT, , kleidicv_gray_to_rgba_u8(u8a, W, rgba, W*4, W, H));
  BENCH("rgb_to_bgr_u8", IT, , kleidicv_rgb_to_bgr_u8(rgb, W*3, bgr, W*3, W, H));
  BENCH("rgba_to_bgra_u8", IT, , kleidicv_rgba_to_bgra_u8(rgba, W*4, rgba, W*4, W, H));
  BENCH("f32_to_u8", IT, , kleidicv_f32_to_u8(fa, W*4, u8c, W, W, H));
  BENCH("f32_to_s8", IT, , kleidicv_f32_to_s8(fa, W*4, s8c, W, W, H));
  BENCH("u8_to_f32", IT, , kleidicv_u8_to_f32(u8a, W, fc, W*4, W, H));
  BENCH("s8_to_f32", IT, , kleidicv_s8_to_f32(s8a, W, fc, W*4, W, H));

  uint8_t *ch0 = (uint8_t *)malloc(N);
  uint8_t *ch1 = (uint8_t *)malloc(N);
  uint8_t *ch2 = (uint8_t *)malloc(N);
  void *split_dst[3] = {ch0, ch1, ch2};
  size_t split_str[3] = {W, W, W};
  const void *merge_src[3] = {ch0, ch1, ch2};
  size_t merge_str[3] = {W, W, W};
  BENCH("split_3ch_u8", IT, , kleidicv_split(rgb, W*3, split_dst, split_str, W, H, 3, 1));
  BENCH("merge_3ch_u8", IT, , kleidicv_merge(merge_src, merge_str, rgb, W*3, W, H, 3, 1));

  // --- YUV ---
  printf("\n=== YUV Conversions ===\n");
  uint8_t *yuv444 = (uint8_t *)malloc(N * 3);
  memset(yuv444, 128, N * 3);
  BENCH("yuv444_to_rgb", IT, , kleidicv_yuv_to_rgb_u8(yuv444, W*3, rgb, W*3, W, H, KLEIDICV_YUV444_TO_RGB));
  BENCH("rgb_to_yuv444", IT, , kleidicv_rgb_to_yuv_u8(rgb, W*3, yuv444, W*3, W, H, KLEIDICV_RGB_TO_YUV444));

  uint8_t *yplane = (uint8_t *)malloc(N);
  uint8_t *uvplane = (uint8_t *)malloc(N / 2);
  memset(yplane, 128, N);
  memset(uvplane, 128, N / 2);
  BENCH("yuv420sp_to_rgb (NV12)", IT, , kleidicv_yuv_semiplanar_to_rgb_u8(yplane, W, uvplane, W, rgb, W*3, W, H, KLEIDICV_NV12_TO_RGB));

  uint8_t *yuv422 = (uint8_t *)malloc(N * 2);
  memset(yuv422, 128, N * 2);
  BENCH("yuv422_to_rgb (YUYV)", IT, , kleidicv_yuv_to_rgb_u8(yuv422, W*2, rgb, W*3, W, H, KLEIDICV_YUYV_TO_RGB));

  uint8_t *yuv420p = (uint8_t *)malloc(N * 3 / 2);
  memset(yuv420p, 128, N * 3 / 2);
  BENCH("yuv420p_to_rgb (IYUV)", IT, , kleidicv_yuv_to_rgb_u8(yuv420p, W, rgb, W*3, W, H, KLEIDICV_IYUV_TO_RGB));

  // --- Analysis ---
  printf("\n=== Analysis ===\n");
  uint8_t mn8, mx8;
  float mnf, mxf, sumf;
  size_t min_off, max_off, cnt;
  BENCH("min_max_u8", IT, , kleidicv_min_max_u8(u8a, W, W, H, &mn8, &mx8));
  BENCH("min_max_s16", IT, , kleidicv_min_max_s16(s16a, W*2, W, H, &s16a[0], &s16a[1]));
  BENCH("min_max_s32", IT, , kleidicv_min_max_s32(s32a, W*4, W, H, &s32a[0], &s32a[1]));
  BENCH("min_max_f32", IT, , kleidicv_min_max_f32(fa, W*4, W, H, &mnf, &mxf));
  BENCH("min_max_loc_u8", IT, , kleidicv_min_max_loc_u8(u8a, W, W, H, &min_off, &max_off));
  BENCH("sum_f32", IT, , kleidicv_sum_f32(fa, W*4, W, H, &sumf));
  BENCH("count_nonzeros_u8", IT, , kleidicv_count_nonzeros_u8(u8a, W, W, H, &cnt));

  // --- Filters ---
  printf("\n=== Filters ===\n");
  uint8_t *blur = (uint8_t *)malloc(N);
  BENCH("gaussian_blur_3x3", IT, , kleidicv_gaussian_blur_u8(u8a, W, blur, W, W, H, 1, 3, 3, 0, 0, KLEIDICV_BORDER_TYPE_REPLICATE));
  BENCH("gaussian_blur_5x5", IT, , kleidicv_gaussian_blur_u8(u8a, W, blur, W, W, H, 1, 5, 5, 0, 0, KLEIDICV_BORDER_TYPE_REPLICATE));
  BENCH("gaussian_blur_7x7", IT, , kleidicv_gaussian_blur_u8(u8a, W, blur, W, W, H, 1, 7, 7, 0, 0, KLEIDICV_BORDER_TYPE_REPLICATE));

  int16_t *s16dst = (int16_t *)malloc(N * 2);
  BENCH("sobel_horiz_u8", IT, , kleidicv_sobel_3x3_horizontal_s16_u8(u8a, W, s16dst, W*2, W, H, 1));
  BENCH("sobel_vert_u8", IT, , kleidicv_sobel_3x3_vertical_s16_u8(u8a, W, s16dst, W*2, W, H, 1));

  int16_t *scharr_dst = (int16_t *)malloc(N * 4);
  BENCH("scharr_u8", IT, , kleidicv_scharr_interleaved_s16_u8(u8a, W, W, H, 1, scharr_dst, (W-2)*4));

  // --- Morphology ---
  printf("\n=== Morphology ===\n");
  BENCH("dilate_3x3_u8", IT, , kleidicv_dilate_u8(u8a, W, blur, W, W, H, 1, 3, 3, 1, 1, KLEIDICV_BORDER_TYPE_REPLICATE, nullptr, 1));
  BENCH("erode_3x3_u8", IT, , kleidicv_erode_u8(u8a, W, blur, W, W, H, 1, 3, 3, 1, 1, KLEIDICV_BORDER_TYPE_REPLICATE, nullptr, 1));

  // --- Resize ---
  printf("\n=== Resize ===\n");
  // resize_to_quarter is an internal API called through resize_linear.
  // We benchmark resize_linear 2x2 upscale instead.
  uint8_t *resize_dst = (uint8_t *)malloc(N * 4);
  BENCH("resize_linear_2x2_u8", IT, , kleidicv_resize_linear_u8(u8a, W, W, H, resize_dst, W*2, W*2, H*2, 1));

  // --- Transform ---
  printf("\n=== Transform ===\n");
  uint8_t *tdst = (uint8_t *)malloc(N);
  BENCH("transpose_u8", IT, , kleidicv_transpose(u8a, W, tdst, H, W, H, 1));
  BENCH("rotate_90cw_u8", IT, , kleidicv_rotate(u8a, W, W, H, tdst, H, 90, 1));

  // Cleanup
  free(u8a); free(u8b); free(u8c);
  free(s8a); free(s8b); free(s8c);
  free(u16a); free(u16b); free(u16c);
  free(s16a); free(s16b); free(s16c);
  free(s32a); free(s32b); free(s32c);
  free(u32a); free(u32b); free(u32c);
  free(s64a); free(s64b); free(s64c);
  free(fa); free(fb); free(fc);
  free(rgb); free(bgr); free(rgba);
  free(ch0); free(ch1); free(ch2);
  free(yuv444); free(yplane); free(uvplane);
  free(yuv422); free(yuv420p);
  free(blur); free(s16dst); free(scharr_dst);
  free(resize_dst); free(tdst);
  return 0;
}
