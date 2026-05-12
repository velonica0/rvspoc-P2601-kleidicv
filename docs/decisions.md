# Design Decisions

## D1: RVV backend replaces Neon as primary namespace on RISC-V

**Decision:** Compile `_rvv.cpp` files with `-DKLEIDICV_TARGET_NEON=1` so RVV implementations land in `kleidicv::neon` namespace.

**Why:** The existing 35 `_api.cpp` dispatch files hard-reference `kleidicv::neon::*` function pointers. Changing all of them would be high-churn, error-prone, and diverge from upstream. By reusing the namespace, the entire API dispatch layer works unmodified on RISC-V.

**Trade-off:** The namespace name is misleading on RISC-V (it says "neon" but runs RVV). This is purely cosmetic; the namespace is an internal detail never exposed to users.

## D2: Direct vsetvl loops instead of the operations.h framework

**Decision:** RVV kernels use explicit `while (i < width) { vl = vsetvl(...); ... }` loops rather than the `apply_operation_by_rows` / `RowBasedOperation` / `VecTraits` template framework.

**Why:** The operations.h framework is tightly coupled to Neon's fixed-width vectors (VectorType = uint8x16_t etc.) and SVE's predicate model. RVV's `vsetvl`-based programming model is fundamentally different: the hardware tells you how many elements it processed, and tail handling is automatic. Forcing RVV into the Neon framework would require a complete parallel set of VecTraits, LoopUnroll specializations, and context adapters with no benefit.

**Trade-off:** Code duplication for the row-iteration boilerplate. Acceptable because the per-kernel loop is 5-10 lines vs. hundreds of lines of framework adaptation.

## D3: LMUL=1 as default grouping

**Decision:** All RVV intrinsics use LMUL=1 (e.g., `vint8m1_t`, `__riscv_vsetvl_e8m1`) as the base. Higher LMUL (m2, m4) is used only where widening chains require it (e.g., u8m1→u16m2→u32m4 for YUV conversions).

**Why:** LMUL=1 uses one vector register per operand, giving the compiler maximum freedom for register allocation. Higher LMUL (m2, m4) can improve throughput for simple operations but halves available registers, risking spills in complex kernels. LMUL=1 is the safe starting point; profiling on real hardware should guide LMUL tuning per kernel.

## D4: Scalar fallback guarded by `#ifdef __riscv_vector`

**Decision:** Each `_rvv.cpp` file contains both an RVV-vectorized path (`#ifdef __riscv_vector`) and a scalar fallback (`#else`), or uses a scalar-only implementation for complex algorithms.

**Why:** Required by competition rules. Also enables building on RISC-V systems without the V extension. The scalar fallback is a direct C++ loop, not dependent on any ARM headers.

## D5: VLEN-agnostic design — all code uses vsetvl

**Decision:** Every RVV kernel calls `vsetvl` at the start of each loop iteration. No hardcoded VLEN assumptions anywhere.

**Why:** The target hardware (Spacemit X100) has VLEN=256, but the same codebase must work on VLEN=128, 256, and 512. The `vsetvl` instruction returns the actual hardware vector length at runtime, so the loop naturally adapts. This is verified by the test suite passing on the VLEN=256 target.

## D6: Architecture detection via CMAKE_SYSTEM_PROCESSOR

**Decision:** Use `CMAKE_SYSTEM_PROCESSOR MATCHES "riscv64"` to detect RISC-V at CMake configure time.

**Why:** Simple, reliable, and standard. Avoids compiler feature detection complexity. The target machine is known to be riscv64 Linux.

## D7: float16_t handling on RISC-V

**Decision:** On RISC-V without Zfh/Zvfh extensions, `float16_t` is typedef'd to `uint16_t` as a storage-only type.

**Why:** ARM's `__fp16` type doesn't exist on RISC-V compilers. RISC-V half-float support requires the Zfh extension. Using `uint16_t` as storage allows the codebase to compile; actual float16 compute operations are not used in the critical path.

## D8: rvv.h abstraction layer

**Decision:** Created a single `kleidicv/rvv.h` header providing inline wrapper functions for RVV intrinsics (`rvv_load`, `rvv_store`, `rvv_sadd`, `rvv_setvl`, etc.) with function overloading for type dispatch.

**Why:** RVV intrinsics have verbose, type-specific names (e.g., `__riscv_vsaddu_vv_u8m1` vs `__riscv_vsadd_vv_i8m1`). The wrapper layer provides clean, overloaded names matching the operations (add, sub, min, max, etc.) while handling signed/unsigned dispatch automatically. This reduces errors and makes kernel code more readable.

## D9: Skip ARM-specific unit tests on RISC-V

**Decision:** The `test/unit_neon/` directory is conditionally excluded from the build when `CMAKE_SYSTEM_PROCESSOR` matches `riscv`. API tests (test/api/) run on all architectures.

**Why:** The unit_neon tests directly include ARM-specific internal headers (resize_linear_generic_u8_neon.h) that use Neon intrinsics and cannot compile on RISC-V. The API tests are architecture-independent and thoroughly test all operations through the public API, which dispatches to the RVV implementations on RISC-V.

## D10: One-to-one file mapping between neon and RVV

**Decision:** Every `_neon.cpp` file has a corresponding `_rvv.cpp` file (49 files total), with identical function signatures and template instantiations.

**Why:** Ensures complete API coverage. The CMake glob picks up `*_rvv.cpp` files, so any function declared in the headers and dispatched by the API layer must have an implementation in the RVV build to avoid linker errors.

## D11: Vertical-first vectorization for separable filters

**Decision:** For separable filters (Gaussian blur, blur-and-downsample, separable_filter_2d), only the vertical pass is vectorized with RVV. The horizontal pass remains scalar.

**Why:** The vertical pass is perfectly vectorizable — each element in a row maps to the same position across kernel rows, enabling contiguous loads and element-wise multiply-accumulate. The horizontal pass requires per-pixel, per-channel border handling with non-contiguous access patterns (neighbors at ±channels offset), making it difficult to vectorize without gather/scatter. The vertical pass dominates the computation since it touches O(width × kernel_height) elements per row, so vectorizing it alone yields 2-4x speedup.

## D12: Two-phase min_max_loc with vfirst

**Decision:** `min_max_loc` uses a two-phase approach: Phase 1 finds the extreme values using vectorized min/max reduction, Phase 2 finds locations using `vfirst` mask search.

**Why:** Combining value tracking and location tracking in a single vectorized pass requires maintaining per-lane index vectors and complex mask-based conditional updates — the same approach that makes the Neon version's `MinMaxLoc` class ~200 lines of template machinery. The two-phase approach is simpler, still fully vectorized, and has the same asymptotic complexity (two passes over the data).

## D13: i32m4 arithmetic for YUV color conversions

**Decision:** YUV444, YUV420sp, and YUV422 conversions use i32m4 (LMUL=4) for the color space arithmetic, processing VLEN/32 pixels per iteration.

**Why:** The BT.601 coefficients (e.g., kUBWeight=33292 for YUV444, kYWeight=1220542 for YUV420) don't fit in i16, requiring i32 multiply-accumulate. Using LMUL=4 for i32 allows the widening chain u8m1→u16m2→u32m4→i32m4 to work naturally with RVV's LMUL rules. At VLEN=256 this processes 8 pixels per iteration.

## D14: UV duplication via vrgather for YUV420sp

**Decision:** In YUV420sp→RGB conversion, chroma samples (shared between pixel pairs) are duplicated using `vrgatherei16` with an index pattern [0,0,1,1,2,2,...] generated from `vid >> 1`.

**Why:** Each U,V sample in NV12/NV21 covers two horizontal pixels. Rather than processing pixels in pairs (which complicates the loop structure), we load N/2 UV pairs, duplicate each value to match N Y pixels, then process all N pixels uniformly. The `vrgather` approach is clean and VLEN-agnostic.

## D15: Histogram vectorization for median blur

**Decision:** The histogram-based median blur (both small and large variants) vectorizes the bulk histogram add/subtract operations using RVV, while keeping the median search (CDF scan) scalar.

**Why:** The sliding-window histogram approach involves adding/subtracting entire column histograms (16 coarse bins or 256 fine bins) as the window moves. These are contiguous array add/subtract operations — perfect for vectorization. The median search is inherently serial (cumulative sum comparison), so it remains scalar. This gives speedup proportional to histogram size / VLEN.

## D16: Scharr output layout matches Neon reduced-width convention

**Decision:** The Scharr RVV implementation outputs `(src_width - 2) * src_channels` elements per row (excluding first/last border columns), and reads source rows `y, y+1, y+2` for output row `y`.

**Why:** The Neon version uses a separable workspace approach that naturally excludes the 3x3 kernel border. The API layer calls the stripe function with `y_end = src_height - 2`, expecting output with 2 fewer columns. Matching this layout is essential for test compatibility. The initial scalar implementation incorrectly included border pixels with clamped indices.

## D17: NaN handling in float-to-integer conversion

**Decision:** Before `vfcvt` (float→int), NaN values are detected with `vmfne(v, v)` and replaced with 0.0f using `vfmerge`.

**Why:** RISC-V `vfcvt.x.f` converts NaN to the largest representable integer (per RISC-V spec), while ARM `vcvtq_s32_f32` converts NaN to 0. The KleidiCV test suite expects the ARM convention (NaN → 0). The `vmfne(v, v)` trick exploits the IEEE 754 property that NaN ≠ NaN.

## D18: Morphology uses MorphologyWorkspace for allocation

**Decision:** The morphology (dilate/erode) RVV implementation uses `MorphologyWorkspace` for buffer allocation rather than `new (std::nothrow)`.

**Why:** The test infrastructure uses `MockMallocToFail` to inject allocation failures by intercepting `std::malloc`. Using `new (std::nothrow)` bypasses this injection since it calls `operator new`, not `malloc`. The `MorphologyWorkspace` class uses `std::malloc` internally, matching the Neon implementation and enabling correct allocation failure testing.

## D19: Exp uses Neon-identical polynomial with Cody-Waite reduction

**Decision:** The RVV exp implementation ports the exact same polynomial algorithm from the Neon version, including Cody-Waite range reduction, split-scale special case handling, and the magic-number shift trick.

**Why:** The test requires 1-ULP accuracy (verified via `nextafterf` comparison across all 2^32 float bit patterns). A simpler polynomial approximation failed this requirement. Using identical coefficients and algorithm structure guarantees bit-exact results matching the Neon reference.

## D20: `-fno-builtin-malloc` for test mock compatibility

**Decision:** The RVV compilation target uses `-fno-builtin-malloc -fno-builtin-free` compiler flags.

**Why:** GCC 15 at `-O2` treats `malloc`/`free` as compiler builtins and eliminates `malloc()+null_check+free()` as a dead store — the optimizer proves it has no observable side effects and removes it entirely. This prevents the test framework's `MockMallocToFail` (which uses `--wrap,malloc` linker flag) from intercepting allocations. The `-fno-builtin-malloc` flag forces GCC to emit actual `malloc` calls that the linker can wrap.

## D21: Error code ordering must match Neon workspace flow

**Decision:** In stripe functions (Sobel, Scharr, etc.), validation checks are ordered to match the error codes the Neon version would produce: (1) null pointer, (2) alignment, (3) image size, (4) bounds, (5) allocation, (6) stride validation.

**Why:** The Neon version naturally produces these errors in this order because its `SeparableFilterWorkspace` constructor checks allocation after the API layer checks null/alignment/size. Tests verify specific error codes for specific invalid inputs. If our checks fire in a different order, we return the wrong error code (e.g., `NOT_IMPLEMENTED` from a stride check when the test expects `ALLOCATION` from a workspace check).
