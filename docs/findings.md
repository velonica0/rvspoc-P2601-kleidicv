# Debugging Findings

## F1: ARM-specific headers in the include chain

**Issue:** `ctypes.h` unconditionally defines `typedef __fp16 float16_t;` which fails on RISC-V compilers.

**Root cause:** The type `__fp16` is an ARM compiler extension defined by `arm_neon.h`/`arm_sve.h`. On RISC-V GCC, this type does not exist.

**Fix:** Added `#ifdef __riscv` guard in `ctypes.h` to use `_Float16` (with Zfh) or `uint16_t` (storage-only fallback).

## F2: dispatch.h ARM auxval detection

**Issue:** `dispatch.h` uses `getauxval(AT_HWCAP2)` with ARM-specific bit positions to detect SVE2/SME.

**Finding:** The entire detection block is inside `#if KLEIDICV_ENABLE_SME2 || KLEIDICV_ENABLE_SME || KLEIDICV_ENABLE_SVE2`. On RISC-V builds, all three are disabled (0), so the code is never compiled. The `#else` branch provides simple pass-through macros that just assign `neon_impl` directly. No changes needed.

## F3: GCC `-flax-vector-conversions` flag

**Issue:** RISC-V GCC does not recognize or need `-flax-vector-conversions` (an ARM-specific flag for Neon vector type coercion).

**Fix:** Guarded the flag with `if(NOT KLEIDICV_IS_RISCV)` in CMakeLists.txt.

## F4: KLEIDICV_STREAMING macro on RISC-V

**Finding:** The `KLEIDICV_STREAMING` macro (used for ARM SME streaming mode) expands to nothing when `KLEIDICV_TARGET_SME` and `KLEIDICV_TARGET_SME2` are both 0. This is the case on RISC-V, so all functions decorated with `KLEIDICV_STREAMING` compile without issues.

## F5: RVV saturating arithmetic intrinsic naming

**Finding:** RVV uses separate intrinsics for signed vs unsigned saturating operations:
- Signed: `__riscv_vsadd_vv_i8m1` (saturating add)
- Unsigned: `__riscv_vsaddu_vv_u8m1` (note the 'u' suffix)

This differs from ARM Neon where `vqaddq` is overloaded for both signed and unsigned types. The `rvv.h` abstraction layer provides unified `rvv_sadd` overloads that dispatch to the correct intrinsic.

## F6: RVV vsetvl tail handling

**Finding:** RVV's `vsetvl` naturally handles the tail of a vector loop. When `remaining_elements < VLEN/SEW`, `vsetvl` returns the smaller count and the hardware processes only those elements. This eliminates the need for separate scalar tail loops, simplifying kernel code compared to the Neon approach (which needs LoopUnroll + scalar_path).

## F7: types.h ARM SME include guard

**Finding:** `types.h` has `#if KLEIDICV_TARGET_SME || KLEIDICV_TARGET_SME2` before `#include <arm_sme.h>`. Since both are 0 on RISC-V, the ARM header is never included. No changes needed.

## F8: Separable filter _sc.h headers use SVE2 intrinsics

**Finding:** The `_sc.h` (scalar) headers in `include/kleidicv/filters/` are not truly portable scalar implementations. They use SVE2 intrinsics (`svld1`, `svst1`, `svwhilelt`, etc.) throughout. The "_sc" suffix appears to mean "SVE-compatible" rather than "scalar-only". This means they cannot be compiled or used on RISC-V. Each filter required a fresh scalar C++ implementation from scratch.

## F9: Template instantiation mismatches cause linker errors

**Finding:** Each `_rvv.cpp` file must instantiate templates for exactly the same types as the corresponding `_neon.cpp` file. Missing instantiations cause undefined reference errors during linking because the `_api.cpp` files (compiled separately) reference specific template specializations.

## F10: Fractional LMUL (mf2, mf4) for widening chains

**Finding:** When vectorizing operations that widen u8->u16->u32, the starting LMUL must be chosen so the widened result stays within legal LMUL bounds. For example, if the output needs i32m4, the input u8 load should use LMUL=m1 (u8m1->u16m2->u32m4). For i16m1 output from u8 input, use u8mf2 (half-LMUL). The `vsetvl` is set for the widest type in the chain.

## F11: RVV reduction intrinsics require an initial scalar vector

**Finding:** RVV's `vredmin`/`vredmax`/`vredsum` reductions take a second vector operand as the initial scalar accumulator (element 0 of a vector). This differs from ARM Neon's `vminvq`/`vmaxvq` which return a scalar directly. The result is placed in element 0 of the output vector, then extracted with `vmv_x_s` or `vfmv_f_s`.

## F12: Segment load/store for multi-channel data

**Finding:** RVV's `vlseg2/3/4e8` and `vsseg2/3/4e8` intrinsics are the natural way to deinterleave/interleave multi-channel image data (RGB, YUV, etc.). They replace ARM Neon's `vld3q`/`vst3q`. The tuple types (`vuint8m1x3_t`) are created with `vcreate` and accessed with `vget`. These work correctly with variable `vl` from `vsetvl`.

## F13: vfirst for vectorized search

**Finding:** `__riscv_vfirst_m_b8(mask, vl)` returns the index of the first set bit in a vector mask, or -1 if none are set. This enables vectorized linear search (e.g., finding the location of a min/max value) without falling back to scalar element-by-element comparison. Combined with `vmseq` (compare equal), it's equivalent to a vectorized `memchr`.

## F14: Strided loads/stores for non-contiguous access

**Finding:** `__riscv_vlse8_v_u8m1(base, stride, vl)` loads `vl` elements with a byte stride between consecutive elements. This is ideal for downsampling operations (e.g., resize_to_quarter picking every 2nd pixel with stride=2) and transpose (strided store with stride=dst_stride). ARM Neon lacks a general strided load — the equivalent requires `vld1q_lane` or table lookup.

## F15: u16 accumulation is safe for normalized convolution kernels

**Finding:** For convolution kernels where the weights sum to a power of 2 (e.g., binomial [1,4,6,4,1] sums to 16, used twice for separable: 16x16=256), the maximum accumulator value for u8 input is 255 x (sum_of_weights) = 255 x 256 = 65280, which fits in u16 (max 65535). This avoids the need for u32 accumulators in the vertical pass, saving register pressure and enabling mf2/m1 LMUL groupings.

## F16: vrgatherei16 EEW ratio requirement

**Finding:** `vrgatherei16` for u8m1 data requires a u16m2 index vector (EEW ratio = 2), not u16m1. This is because the index element width (16-bit) is 2x the data element width (8-bit), so the index LMUL must also be 2x. Using u16m1 causes a GCC type error.

## F17: kleidicv.h has `#ifndef __aarch64__` guard

**Finding:** The upstream `kleidicv.h` header has `#ifndef __aarch64__ / #error` on line 94, and `KLEIDICV_MAX_IMAGE_PIXELS` is guarded by `#ifdef __aarch64__`. Both must be patched for RISC-V: the error guard extended with `&& !defined(__riscv)`, and the macro definition moved outside the guard.

## F18: GCC 15 segment load/store tuple API

**Finding:** GCC 15 uses the newer RVV intrinsic API for segment loads/stores. The old API `__riscv_vlseg3e8_v_u8m1(&r, &g, &b, ptr, vl)` with output pointers is replaced by `__riscv_vlseg3e8_v_u8m1x3(ptr, vl)` returning a tuple type. Elements are extracted with `__riscv_vget_v_u8m1x3_u8m1(tuple, idx)` and tuples created with `__riscv_vcreate_v_u8m1x3(v0, v1, v2)`.

## F19: resize_to_quarter does 2x2 averaging, not 4x4

**Finding:** The KleidiCV `resize_to_quarter_u8` function halves both dimensions using 2x2 block averaging (each output pixel = average of a 2x2 block), NOT 4x4 as the name "quarter" might suggest. Output dimensions are src_width/2 x src_height/2. The "quarter" refers to the output area being 1/4 of the input.

## F20: int32 absdiff overflows with max(a-b, b-a)

**Finding:** Computing `|a-b|` for int32 as `max(a-b, b-a)` causes signed overflow when `a` and `b` have opposite signs and large magnitudes (e.g., INT_MAX and -1). The correct approach widens to int64 first: `vsext_vf2` -> subtract -> negate -> max -> narrow with saturating clip.

## F21: RISC-V vfcvt maps NaN to INT_MAX, not 0

**Finding:** RISC-V's `vfcvt.x.f` converts NaN to the largest representable integer (per RISC-V F extension spec), while ARM's `vcvtq_s32_f32` converts NaN to 0. This causes float-to-int8/uint8 conversions to produce 127/255 instead of 0 for NaN inputs. Fix: detect NaN with `vmfne(v, v)` (NaN != NaN) and replace with 0.0f before conversion.

## F22: Scharr output layout mismatch

**Finding:** The Neon Scharr implementation uses a separable workspace approach that produces output with reduced dimensions: `(src_width - 2) * src_channels` columns and `src_height - 2` rows. The API layer calls the stripe function with `y_end = src_height - 2`. The initial scalar implementation incorrectly wrote output for all input pixels including borders, causing value mismatches and buffer overflows. Fix: map output position ox to source position ox + src_channels (skipping first column).

## F23: Morphology allocation via std::malloc, not operator new

**Finding:** The test framework uses `MockMallocToFail` to inject allocation failures by intercepting `std::malloc`. The morphology implementation must use `std::malloc` (via `MorphologyWorkspace`) for its temp buffers, not `new (std::nothrow)` which calls `operator new` and bypasses the mock. Using the wrong allocator causes the CannotAllocateImage test to fail (function succeeds when it should return KLEIDICV_ERROR_ALLOCATION).

## F24: Spacemit X100 has VLEN=256

**Finding:** The target RISC-V board (Spacemit X100) has VLEN=256 bits (32 bytes per vector register). This was confirmed by running `vsetvl_e8m1(65536)` which returns 32. The test framework reports "Vector length is set to 16 bytes" which is a Neon-centric default, not the actual hardware VLEN.

## F25: GCC 15 -O2 eliminates malloc+free as dead store

**Finding:** GCC 15 with `-O2` recognizes `malloc`/`free` as builtins and applies Dead Store Elimination to sequences like `void *p = malloc(N); if (!p) return ERR; free(p);`. The compiler proves the allocation has no observable side effect and removes the entire sequence. This is invisible in normal code but breaks the test framework's `MockMallocToFail` mechanism, which uses `--wrap,malloc` to intercept allocation calls. Fix: compile with `-fno-builtin-malloc -fno-builtin-free` to force GCC to emit actual function calls.

## F26: Remap float-to-int cast is undefined behavior for out-of-range values

**Finding:** The remap implementation converts float map coordinates to integer pixel positions via `static_cast<int>(float_val)`. When the float value is `inf`, `NaN`, or outside `[INT_MIN, INT_MAX]`, this cast is undefined behavior in C++. On RISC-V GCC 15, this produces unpredictable values causing incorrect pixel lookups. Fix: clamp via `double` to `[INT_MIN, INT_MAX]` before casting, or use `std::lround` with bounds checking.

## F27: Generic resize coefficient rounding must match Neon fixed-point

**Finding:** The Neon generic resize (`resize_linear_generic_neon.cpp`) uses a specific fixed-point coordinate computation: `aligned_scale` with 16-bit fractional precision, center-aligned origin, and 1/512 rounding bias. The interpolation uses `(a<<8 + (b-a)*frac + 128) >> 8` pattern (matching Neon's `vraddhn_u16`). A scalar implementation using floating-point arithmetic produces different rounding at certain pixel positions, causing 1-3 value differences in ~0.001% of pixels. Fix: replicate the exact fixed-point formula.
