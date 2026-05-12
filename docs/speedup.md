# Speedup: RVV vs Scalar

Target: Spacemit X100 (rv64gcv, VLEN=256, 8 cores), GCC 15.2, `-O2`

Both builds use the same source code (`_rvv.cpp` files). The RVV build compiles with `-march=rv64gcv` (activating `#ifdef __riscv_vector` path). The scalar build compiles with `-march=rv64gc` (activating `#else` scalar fallback path). Same API, same tests, same binary — only the compiled code path differs.

## Measured Results (1920x1080, 100 iterations)

### Arithmetic

| Operation | Scalar (ms) | RVV (ms) | Speedup |
|-----------|------------|---------|---------|
| saturating_add_u8 | 3.090 | 0.633 | 4.9x |
| saturating_add_s8 | 4.688 | 0.629 | 7.5x |
| saturating_add_u16 | 3.518 | 1.212 | 2.9x |
| saturating_add_s16 | 4.186 | 1.242 | 3.4x |
| saturating_add_u32 | 3.443 | 2.331 | 1.5x |
| saturating_add_s32 | 5.026 | 2.399 | 2.1x |
| saturating_add_s64 | 5.520 | 5.147 | 1.1x |
| saturating_sub_u8 | 3.794 | 0.628 | 6.0x |
| saturating_absdiff_u8 | 3.431 | 0.637 | 5.4x |
| saturating_absdiff_s8 | 3.657 | 0.648 | 5.6x |
| saturating_absdiff_s32 | 3.805 | 2.578 | 1.5x |
| bitwise_and_u8 | 2.844 | 0.627 | 4.5x |
| compare_equal_u8 | 2.847 | 0.638 | 4.5x |
| compare_greater_u8 | 2.847 | 0.638 | 4.5x |
| threshold_binary_u8 | 2.840 | 0.391 | 7.3x |
| in_range_u8 | 2.884 | 0.419 | 6.9x |
| scale_u8 | 1.902 | 1.903 | 1.0x |
| multiply_u8 | 4.944 | 0.567 | 8.7x |
| multiply_s16 | 5.546 | 1.169 | 4.7x |
| add_abs_thresh_s16 | 7.602 | 1.659 | 4.6x |
| exp_f32 | 52.881 | 6.511 | 8.1x |

### Conversions

| Operation | Scalar (ms) | RVV (ms) | Speedup |
|-----------|------------|---------|---------|
| gray_to_rgb_u8 | 2.855 | 0.454 | 6.3x |
| gray_to_rgba_u8 | 3.866 | 0.562 | 6.9x |
| rgb_to_bgr_u8 | 2.992 | 0.833 | 3.6x |
| rgba_to_bgra_u8 | 3.984 | 1.126 | 3.5x |
| f32_to_u8 | 9.839 | 1.386 | 7.1x |
| f32_to_s8 | 9.947 | 1.419 | 7.0x |
| u8_to_f32 | 1.909 | 0.605 | 3.2x |
| s8_to_f32 | 1.909 | 0.607 | 3.1x |
| split_3ch_u8 | 17.674 | 0.844 | 20.9x |
| merge_3ch_u8 | 18.855 | 0.984 | 19.2x |

### YUV Conversions

| Operation | Scalar (ms) | RVV (ms) | Speedup |
|-----------|------------|---------|---------|
| yuv444_to_rgb | 14.178 | 4.908 | 2.9x |
| rgb_to_yuv444 | 10.328 | 5.141 | 2.0x |
| yuv420sp_to_rgb (NV12) | 18.180 | 4.474 | 4.1x |
| yuv422_to_rgb (YUYV) | 18.765 | 4.920 | 3.8x |
| yuv420p_to_rgb (IYUV) | 18.691 | 18.076 | 1.0x |

### Analysis

| Operation | Scalar (ms) | RVV (ms) | Speedup |
|-----------|------------|---------|---------|
| min_max_u8 | 3.850 | 0.245 | 15.7x |
| min_max_s16 | 3.792 | 0.492 | 7.7x |
| min_max_s32 | 3.792 | 1.006 | 3.8x |
| min_max_f32 | 5.683 | 1.133 | 5.0x |
| min_max_loc_u8 | 3.784 | 0.245 | 15.4x |
| sum_f32 | 2.847 | 3.327 | 0.9x |
| count_nonzeros_u8 | 1.895 | 0.447 | 4.2x |

### Filters

| Operation | Scalar (ms) | RVV (ms) | Speedup |
|-----------|------------|---------|---------|
| gaussian_blur_3x3 | 42.970 | 23.540 | 1.8x |
| gaussian_blur_5x5 | 63.136 | 31.119 | 2.0x |
| gaussian_blur_7x7 | 82.927 | 39.080 | 2.1x |
| sobel_horiz_u8 | 8.551 | 1.606 | 5.3x |
| sobel_vert_u8 | 8.515 | 1.601 | 5.3x |
| scharr_u8 | 12.568 | 2.426 | 5.2x |

### Morphology

| Operation | Scalar (ms) | RVV (ms) | Speedup |
|-----------|------------|---------|---------|
| dilate_3x3_u8 | 27.175 | 1.082 | 25.1x |
| erode_3x3_u8 | 28.010 | 1.078 | 26.0x |

### Resize

| Operation | Scalar (ms) | RVV (ms) | Speedup |
|-----------|------------|---------|---------|
| resize_linear_2x2_u8 | 11.052 | 11.081 | 1.0x |

### Transform

| Operation | Scalar (ms) | RVV (ms) | Speedup |
|-----------|------------|---------|---------|
| transpose_u8 | 5.935 | 5.740 | 1.0x |
| rotate_90cw_u8 | 5.953 | 5.743 | 1.0x |

## Summary

| Speedup Range | Operators |
|--------------|-----------|
| 15-26x | min_max_u8, min_max_loc_u8, split_3ch, merge_3ch, dilate, erode |
| 5-9x | add_s8, sub_u8, absdiff, threshold, in_range, multiply, exp, gray_to_rgb/rgba, f32_to_u8/s8, sobel, scharr |
| 2-5x | add_u16/s16, compare, count_nonzeros, yuv444, yuv420sp, yuv422, gaussian_blur, min_max_s16/f32, rgb_to_bgr, u8_to_f32 |
| ~1x (no speedup) | scale_u8, add_u32/s64, yuv420p (scalar-only), resize_linear, transpose, rotate, sum_f32 |

**Geometric mean speedup across all RVV-vectorized operators: ~4.5x**

Notable observations:
- **split/merge** achieve 20x because the scalar path has poor cache behavior (column-by-column access), while RVV uses segment load/store which deinterleaves in hardware.
- **dilate/erode** achieve 25x because the scalar path iterates per-pixel with nested kernel loops, while RVV loads entire rows and applies element-wise min/max.
- **scale_u8** shows no speedup because the scalar path is already optimized by GCC's auto-vectorizer.
- **sum_f32** is slightly slower with RVV due to the `vfredusum` ordered reduction being serial, while the scalar loop benefits from GCC's aggressive FP optimization.
- **transpose/rotate** are memory-bound (strided stores) and see no measurable improvement.

## Reproduction

```bash
ssh openkylin@192.168.5.211
cd /home/openkylin/github/rvspoc-P2601-kleidicv
bash scripts/run_bench.sh
```

Or manually:
```bash
# RVV
c++ -O2 -std=c++17 -Wno-unused-result \
    -Ikleidicv/include -Ibuild/kleidicv/include \
    -o /tmp/bench_rvv scripts/bench.cpp -Lbuild/kleidicv -lkleidicv
/tmp/bench_rvv

# Scalar
c++ -O2 -std=c++17 -march=rv64gc -Wno-unused-result \
    -Ikleidicv/include -Ibuild_scalar/kleidicv/include \
    -o /tmp/bench_scalar scripts/bench.cpp -Lbuild_scalar/kleidicv -lkleidicv
/tmp/bench_scalar
```

## On-Device Test Results

**Target:** Spacemit X100, rv64gcv, VLEN=256, GCC 15.2

| Build | Total | Passed | Failed | Skipped | Crashed |
|-------|-------|--------|--------|---------|---------|
| RVV (rv64gcv) | 4543 | 4526 | 0 | 17 | 0 |
| Scalar (rv64gc) | 4543 | 4526 | 0 | 17 | 0 |

Both builds produce identical test results. The 17 skipped tests are intentionally long-running (pass with `--long-running-tests`).
