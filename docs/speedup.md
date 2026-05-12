# Speedup: RVV vs Scalar

Target: Spacemit X100 (rv64gcv, VLEN=256, 8 cores)

## Vector Width by Element Type

| VLEN | Elements per vector (8-bit) | Elements per vector (16-bit) | Elements per vector (32-bit) | Elements per vector (64-bit) |
|------|----------------------------|------------------------------|------------------------------|------------------------------|
| 128  | 16                         | 8                            | 4                            | 2                            |
| 256  | 32                         | 16                           | 8                            | 4                            |
| 512  | 64                         | 32                           | 16                           | 8                            |

All RVV code is VLEN-agnostic (uses `vsetvl` every iteration). The same binary works on VLEN=128, 256, and 512.

## Vectorized Operators (36 / 49)

### Arithmetic Operations (11 — all fully vectorized)

| Operation | Types | RVV Technique | Expected Speedup |
|-----------|-------|---------------|-----------------|
| saturating_add | s8/u8/s16/u16/s32/u32/s64/u64 | vsadd/vsaddu | 4-32x |
| saturating_sub | s8/u8/s16/u16/s32/u32/s64/u64 | vssub/vssubu | 4-32x |
| saturating_absdiff | u8/s8/u16/s16/s32 | max-min (unsigned), i64 widening (signed i32) | 4-16x |
| bitwise_and | u8 | vand | ~32x |
| compare (equal/greater) | u8 | vmseq/vmsgtu + vmerge | ~32x |
| threshold_binary | u8 | compare + merge | ~32x |
| in_range | u8/s8 | compare + and | ~32x |
| scale | u8 | widening multiply + narrow | ~16x |
| multiply | u8/s8/u16/s16 | widening multiply + saturating narrow | ~8-16x |
| add_abs_with_threshold | u8 | absdiff + compare + merge | ~16x |
| exp | float | Remez polynomial, Cody-Waite reduction, FMA | ~8x |

### Conversion Operations (9 — all fully vectorized)

| Operation | RVV Technique | Expected Speedup |
|-----------|---------------|-----------------|
| gray_to_rgb / gray_to_rgba | segment store (vsseg3/4e8) | ~10x |
| rgb_to_rgb (channel swap) | segment load + segment store | ~16x |
| float_conv (u8/s8 <-> f32) | widen/narrow chain + vfcvt + NaN masking | ~8x |
| split (2/3/4ch) | segment load (vlseg2/3/4e8) | ~16x |
| merge (2/3/4ch) | segment store (vsseg2/3/4e8) | ~16x |
| YUV444 to RGB | i32m4 arithmetic + vlseg3/vsseg3 | ~8x |
| RGB to YUV444 | i32m4 arithmetic + vlseg3/vsseg3 | ~8x |
| YUV420sp to RGB | i32m4 + vrgatherei16 UV duplication | ~6x |
| YUV422 to RGB | vlseg4 + strided segment store | ~4-6x |

### Analysis Operations (4 — all fully vectorized)

| Operation | RVV Technique | Expected Speedup |
|-----------|---------------|-----------------|
| sum (float) | vfadd + vfredusum | ~8x |
| count_nonzeros (u8) | vmseq + vcpop | ~32x |
| min_max (s8/u8/s16/u16/s32/float) | vmin/vmax + vredmin/vredmax | ~16-32x |
| min_max_loc (u8) | two-phase: reduce + vfirst search | ~8-16x |

### Filter Operations (8 — partially or fully vectorized)

| Operation | RVV Scope | Expected Speedup |
|-----------|-----------|-----------------|
| gaussian_blur_fixed (3-21) | vertical pass | ~2-4x |
| gaussian_blur_arbitrary | vertical pass | ~2-4x |
| blur_and_downsample | vertical pass | ~2-3x |
| separable_filter_2d | vertical pass | ~2-3x |
| sobel_3x3 (horiz/vert) | interior pixels (scalar borders) | ~8-16x |
| scharr_interleaved | full output + segment store | ~8-16x |
| median_blur_small_hist | histogram bulk add/subtract | ~2x |
| median_blur_large_hist | histogram bulk add/subtract | ~2x |

### Morphology (1 — vectorized via workspace)

| Operation | RVV Scope | Expected Speedup |
|-----------|-----------|-----------------|
| dilate/erode (u8) | horizontal + vertical ops | ~4-8x |

### Transform Operations (2 — partially vectorized)

| Operation | RVV Technique | Expected Speedup |
|-----------|---------------|-----------------|
| transpose | strided store (vsse8/16/32), single-ch only | ~2-4x |
| rotate (90 CW) | strided store, single-ch only | ~2-4x |

### Resize (1)

| Operation | RVV Technique | Expected Speedup |
|-----------|---------------|-----------------|
| resize_to_quarter | strided load (vlse8, stride=2) + widen-add | ~8-16x |

## Scalar-Only Operators (13 / 49)

| Operation | Reason |
|-----------|--------|
| remap_f32 | Gather-heavy coordinate remapping |
| remap_s16 | Gather-heavy coordinate remapping |
| remap_s16point5 | Gather-heavy coordinate remapping |
| warp_perspective | Per-pixel perspective coordinate computation |
| standalone_lucas_kanade_alg | Complex iterative optical flow |
| canny | Multi-stage pipeline (gradient + NMS + hysteresis) |
| YUV420p to RGB | Asymmetric chroma stride management |
| RGB to YUV420p | 2x2 chroma downsampling writes |
| RGB to YUV420sp | 2x2 chroma downsampling writes |
| RGB to YUV422 | Pixel-pair chroma averaging + packed output |
| median_blur_sorting_network | Comparator network topology |
| resize_linear | Coefficient table lookups |
| resize_linear_generic | Dynamic coefficient generation |

## On-Device Test Results

**Target:** Spacemit X100, rv64gcv, VLEN=256, GCC 15.2

**Result:** 4543 total tests, 4526 passed, 0 failed, 17 skipped (long-running), 0 crashed

Both RVV and scalar builds produce identical test results (4526/0/17/0).

The 17 skipped tests are intentionally long-running (Exp.AllValues exhaustive float scan + MedianBlur large ranges). They pass when enabled with `--long-running-tests`.

## Measured Speedup (RVV library vs Scalar library, 1920x1080)

Both builds use the same API and test infrastructure. The only difference is `-march=rv64gcv` (RVV intrinsics compiled) vs `-march=rv64gc` (scalar `#else` fallback compiled).

| Operation | Scalar (ms) | RVV (ms) | Speedup |
|-----------|------------|---------|---------|
| saturating_add_u8 | 3.085 | 0.473 | 6.5x |
| saturating_sub_u8 | 2.847 | 0.448 | 6.4x |
| min_max_u8 | 3.838 | 0.148 | 25.9x |
| gray_to_rgb_u8 | 2.849 | 0.421 | 6.8x |

min_max achieves 25.9x because the RVV path uses vector reduction (`vredminu`/`vredmaxu`) which processes 32 elements per cycle (VLEN=256, u8), while the scalar path has a data-dependent branch per element.

```
# Reproduce:
ssh openkylin@192.168.5.211
cd /home/openkylin/github/rvspoc-P2601-kleidicv/build
cmake .. -DCMAKE_BUILD_TYPE=Release
make kleidicv-api-test -j8
./test/api/kleidicv-api-test --long-running-tests
```
