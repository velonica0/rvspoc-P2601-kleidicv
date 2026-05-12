# Unoptimized Operators

Operators that currently use scalar fallback on RISC-V and could benefit from RVV vectorization.

## Remaining Scalar-Only Operators (13 / 49)

### Complex algorithms (gather-heavy or iterative)

- [ ] **remap_f32** - Float32 coordinate remap with interpolation
- [ ] **remap_s16** - Int16 coordinate remap
- [ ] **remap_s16point5** - Fixed-point coordinate remap
- [ ] **warp_perspective** - Perspective transformation with interpolation
- [ ] **standalone_lucas_kanade_alg** - Optical flow (iterative with bilinear resampling)
- [ ] **canny** - Canny edge detection (gradient + NMS + hysteresis pipeline)

### Subsampled YUV encoding (2x2 chroma downsampling writes)

- [ ] **YUV420p to RGB** - Planar 4:2:0 with asymmetric chroma stride
- [ ] **RGB to YUV420p** - 2x2 chroma downsampling to planar output
- [ ] **RGB to YUV420sp** - 2x2 chroma downsampling to semi-planar output
- [ ] **RGB to YUV422** - Pixel-pair chroma averaging to packed output

### Other

- [ ] **median_blur (sorting_network)** - Small kernel median via comparator networks
- [ ] **resize_linear** - Bilinear interpolation with coefficient table lookups
- [ ] **resize_linear_generic** - Generic resize with dynamic coefficient generation

## Already Vectorized with RVV (36 / 49)

### Arithmetic (11 — all vectorized)
- [x] saturating_add (s8/u8/s16/u16/s32/u32/s64/u64)
- [x] saturating_sub (s8/u8/s16/u16/s32/u32/s64/u64)
- [x] saturating_absdiff (u8/s8/u16/s16/s32 — int32 uses i64 widening)
- [x] bitwise_and (u8)
- [x] compare equal/greater (u8)
- [x] threshold_binary (u8)
- [x] in_range (u8/s8)
- [x] scale (u8)
- [x] multiply (u8/s8/u16/s16)
- [x] add_abs_with_threshold (u8)
- [x] exp (float — Remez polynomial with Cody-Waite reduction, 1-ULP accurate)

### Conversions (9 — all vectorized)
- [x] gray_to_rgb / gray_to_rgba (segment store)
- [x] rgb_to_rgb (channel swap, segment load/store)
- [x] float_conv (u8<->f32, s8<->f32 — widen/narrow + NaN masking)
- [x] split (2/3/4 channel deinterleave)
- [x] merge (2/3/4 channel interleave)
- [x] YUV444 to RGB (i32m4 arithmetic, all BGR/RGB/BGRA/RGBA variants)
- [x] RGB to YUV444 (i32m4 arithmetic, all variants)
- [x] YUV420sp to RGB (NV12/NV21 with vrgather UV duplication)
- [x] YUV422 to RGB (vlseg4 + strided segment store, all YUYV/UYVY/YVYU variants)

### Analysis (4 — all vectorized)
- [x] sum (float reduction)
- [x] count_nonzeros (u8 mask popcount)
- [x] min_max (s8/u8/s16/u16/s32/float — vector reduction)
- [x] min_max_loc (u8 — two-phase: vector reduce + vfirst search)

### Filters (8 — partially or fully vectorized)
- [x] gaussian_blur_fixed (vertical pass: binomial + half-kernel)
- [x] gaussian_blur_arbitrary (vertical pass: half-kernel)
- [x] blur_and_downsample (vertical pass: 5-tap binomial)
- [x] separable_filter_2d (vertical pass: u8 and u16)
- [x] sobel 3x3 (horizontal + vertical, interior RVV + scalar borders)
- [x] scharr interleaved (full output + vsseg2 store)
- [x] median_blur_small_hist (histogram add/subtract)
- [x] median_blur_large_hist (histogram add/subtract)

### Morphology (1)
- [x] dilate/erode (u8 — via MorphologyWorkspace + RVV horizontal/vertical ops)

### Transform (2 — partially vectorized)
- [x] transpose (out-of-place, single-channel u8/u16/u32 via strided store)
- [x] rotate 90 CW (single-channel u8/u16/u32 via strided store)

### Resize (1)
- [x] resize_to_quarter (2x2 block averaging with strided loads)

## Summary

| Category | Vectorized | Scalar-only | Total |
|----------|-----------|-------------|-------|
| Arithmetic | 11 | 0 | 11 |
| Conversions | 9 | 0 | 9 |
| YUV encoding | 0 | 4 | 4 |
| Analysis | 4 | 2 | 6 |
| Filters | 8 | 1 | 9 |
| Morphology | 1 | 0 | 1 |
| Transform | 2 | 4 | 6 |
| Resize | 1 | 2 | 3 |
| **Total** | **36** | **13** | **49** |

## On-Device Test Results (Spacemit X100, VLEN=256)

930 / 930 tests passed (100%) including long-running exhaustive exp test.

All code is VLEN-agnostic — the same binary works on VLEN=128, 256, and 512.
