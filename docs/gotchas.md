# Gotchas and Pitfalls

## G1: Namespace mismatch between RVV code and API dispatch

**Pitfall:** The `_api.cpp` files reference `kleidicv::neon::*` functions. If you compile RVV code into a different namespace (e.g., `kleidicv::rvv`), the linker will fail with undefined references.

**Solution:** Compile `_rvv.cpp` files with `-DKLEIDICV_TARGET_NEON=1` so `KLEIDICV_TARGET_NAMESPACE` resolves to `kleidicv::neon`.

## G2: RVV signed vs unsigned intrinsic variants

**Pitfall:** RVV has distinct intrinsics for signed and unsigned operations. Using the wrong variant silently produces incorrect results:
- `__riscv_vsadd` = signed saturating add
- `__riscv_vsaddu` = unsigned saturating add
- `__riscv_vmin` = signed min
- `__riscv_vminu` = unsigned min

The `rvv.h` wrappers (`rvv_sadd`, `rvv_min`, etc.) handle this via overloading, but if you write raw intrinsics, double-check the signedness suffix.

## G3: LMUL mismatch in widening operations

**Pitfall:** Widening operations (e.g., `__riscv_vwmulu_vv_u16m2`) produce results with 2x LMUL. If you widen from m1, the result is m2. You must use the m2 type for the result and m1 for subsequent narrowing. Mismatched LMUL causes compile errors or silent truncation.

## G4: vsetvl must match the element width of the operation

**Pitfall:** The vector length (`vl`) returned by `vsetvl` is the count of elements at the specified SEW/LMUL. When widening (e.g., u8->u16), the vl is still valid — it counts elements, not bytes. But you must call `vsetvl` with the SEW that determines how many elements you want to process. For example, `vsetvl_e16m1` for i16 output means the u8 input loads should use mf2 LMUL.

## G5: Stride-based row access vs contiguous arrays

**Pitfall:** KleidiCV images use byte strides between rows. The stride may be larger than `width * sizeof(T)` (padding) or even non-contiguous. Always use `row_ptr(base, stride, row)` to compute row pointers. Never assume `src + row * width`.

## G6: float16_t is not a real float on RISC-V without Zfh

**Pitfall:** On RISC-V without the Zfh extension, `float16_t` is typedef'd to `uint16_t`. Any code that performs arithmetic on `float16_t` values will get integer arithmetic instead. Also causes duplicate template instantiations (e.g., `dump<float16_t>` = `dump<uint16_t>`). Guard float16-specific instantiations with `#if !defined(__riscv) || defined(__riscv_zfh)`.

## G7: __riscv_vector vs __riscv feature macros

**Pitfall:** `__riscv` is defined for any RISC-V target. `__riscv_vector` is only defined when the V extension is enabled (e.g., `-march=rv64gcv`). Use `__riscv_vector` to guard RVV intrinsic code, not `__riscv`.

## G8: RVV reduction intrinsic API

**Pitfall:** RVV reduction intrinsics (e.g., `__riscv_vredsum`) require an initial scalar value passed as a single-element vector, not a plain scalar. You must create this with `__riscv_vmv_v_x_*` first. The result is also a single-element vector that must be extracted with `__riscv_vmv_x_s_*`.

## G9: Narrowing clip rounding mode

**Pitfall:** `__riscv_vnclipu_wx` and `__riscv_vnclip_wx` require an explicit rounding mode parameter (`__RISCV_VXRM_RDN`, `__RISCV_VXRM_RNU`, etc.). Forgetting this parameter causes a compile error. Use `__RISCV_VXRM_RDN` (round-down/truncate) to match typical image processing behavior.

## G10: LMUL must match across widening chains

**Pitfall:** When using widening operations like `vzext_vf2`, the output LMUL is 2x the input LMUL. A common mistake: loading u8m1 and trying to widen to u16m1 — the correct result type is u16m2. Similarly, u8mf2 widens to u16m1. The `vsetvl` must be set for the element width at the widest point in the chain.

## G11: vmacc operand order differs from ARM

**Pitfall:** RVV `vmacc` is `vd = vd + vs1 * vs2` (accumulate into the destination). ARM Neon `vmlaq` is `vd = vs1 + vs2 * vs3` (accumulate into an explicit source). The operand order reversal is a common source of bugs when translating Neon code to RVV.

## G12: GCC 15 segment load/store tuple API

**Pitfall:** GCC 15 uses the new tuple-based segment load/store API. The old API `__riscv_vlseg3e8_v_u8m1(&r, &g, &b, ptr, vl)` no longer compiles. Must use `__riscv_vlseg3e8_v_u8m1x3(ptr, vl)` + `__riscv_vget_v_u8m1x3_u8m1(tuple, idx)`. The function name encodes both the tuple type and the element type. GCC suggests the correct name in the error message.

## G13: vrgatherei16 index LMUL must be 2x the data LMUL

**Pitfall:** `vrgatherei16` for u8m1 data requires a u16m2 index vector, not u16m1. The EEW ratio (index width / data width = 16/8 = 2) determines the required LMUL ratio. Using the wrong LMUL causes a type mismatch compile error on GCC 15.

## G14: NaN-to-integer conversion differs between ARM and RISC-V

**Pitfall:** RISC-V `vfcvt.x.f` converts NaN to INT_MAX (or UINT_MAX for unsigned). ARM `vcvtq_s32_f32` converts NaN to 0. If your code converts float->int and the test expects ARM behavior, you must explicitly handle NaN before conversion: `vbool mask = vmfne(v, v); v = vfmerge(v, 0, mask);`

## G15: Scharr/Sobel output dimensions differ from input

**Pitfall:** The Neon Scharr/Sobel implementations produce output with reduced dimensions due to the 3x3 kernel border. For Scharr: output has `(src_width - 2) * channels` columns and `src_height - 2` rows. The stripe function is called with `y_end = src_height - 2`. Source rows for output row y are `y, y+1, y+2`. A direct scalar implementation that uses border-clamped indices for ALL input pixels produces wrong results and buffer overflows.

## G16: Morphology must use std::malloc, not operator new

**Pitfall:** The KleidiCV test framework injects allocation failures via `MockMallocToFail` which wraps `std::malloc`. Using `new (std::nothrow)` or `std::unique_ptr<T[]>(new ...)` bypasses this mock because `operator new` is a separate allocation path. The Neon morphology uses `MorphologyWorkspace` (which calls `malloc`), so the RVV version must do the same to pass the CannotAllocateImage test.

## G17: GCC 15 -O2 eliminates malloc+free — must use -fno-builtin-malloc

**Pitfall:** GCC 15 with `-O2` treats `malloc`/`free` as compiler builtins and eliminates sequences like `p = malloc(N); if (!p) return; free(p);` because it proves the allocation is a dead store with no side effects. This means `--wrap,malloc` (used by the test framework's `MockMallocToFail`) never fires, and allocation failure tests pass when they should fail. The fix is to compile with `-fno-builtin-malloc -fno-builtin-free` so GCC emits actual function calls that the linker can intercept.

## G18: Stripe function bounds and error code ordering

**Pitfall:** Tests verify specific error codes for specific invalid inputs. The Neon version's error code ordering follows from its implementation structure (workspace allocation after pointer/size checks). RVV implementations that add custom stride validation checks must place them AFTER the allocation check to match the expected error code ordering. Example: if a stride check returns `NOT_IMPLEMENTED` before the allocation check, a test expecting `ALLOCATION` for a large width will get the wrong error code.

## G19: static_cast<int>(float) is UB for out-of-range values

**Pitfall:** Converting a `float` to `int` via `static_cast<int>()` is undefined behavior when the float value is outside `[INT_MIN, INT_MAX]`, is `inf`, or is `NaN`. On RISC-V GCC 15 this produces arbitrary values. Remap and warp perspective must clamp float coordinates (via `double`) before the int cast.
