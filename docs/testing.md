# Testing Procedure

## Target Hardware

- **Board:** Spacemit X100
- **ISA:** rv64gcv (RV64 + V extension)
- **VLEN:** 256 bits (32 bytes per vector register)
- **Cores:** 8
- **OS:** openKylin (Linux 6.18.3+)
- **Compiler:** GCC 15.2.0 (Bianbu)

## Build

### Prerequisites

GoogleTest 1.12.1 must be available locally. If no internet access, download on a connected machine and extract:

```bash
tar xzf googletest-release-1.12.1.tar.gz
mv googletest-release-1.12.1 /tmp/gtest
cd /tmp/gtest
patch --strip=1 --input=/path/to/rvspoc-P2601-kleidicv/test/patches/googletest.patch
```

### Configure and Build (RVV)

```bash
cd /path/to/rvspoc-P2601-kleidicv
mkdir -p build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DKLEIDICV_BENCHMARK=OFF \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/tmp/gtest

make kleidicv-api-test -j$(nproc)
```

### Configure and Build (Scalar fallback)

To build the scalar-only version (no RVV intrinsics, uses `#else` fallback in all `_rvv.cpp` files):

```bash
cd /path/to/rvspoc-P2601-kleidicv

# Change rv64gcv -> rv64gc in CMakeLists.txt line 201
sed -i 's/-march=rv64gcv/-march=rv64gc/' kleidicv/CMakeLists.txt

mkdir -p build_scalar && cd build_scalar
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DKLEIDICV_BENCHMARK=OFF \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/tmp/gtest
make kleidicv-api-test -j$(nproc)

# Restore CMakeLists.txt
sed -i 's/-march=rv64gc/-march=rv64gcv/' ../kleidicv/CMakeLists.txt
```

### Build Notes

- The C example (`kleidicv-c-example`) may fail to link due to `-fPIC` issues. This is pre-existing and unrelated to RVV work. Use `make kleidicv-api-test` to build only the test binary.
- The `kleidicv_rvv` target compiles with `-march=rv64gcv -DKLEIDICV_TARGET_NEON=1 -fno-builtin-malloc -fno-builtin-free`.
- The `-fno-builtin-malloc` flag is required because GCC 15 at `-O2` eliminates `malloc()+free()` as a dead store, which breaks the test framework's `MockMallocToFail` mechanism.

## Run Tests

### Full test suite (default, ~2 minutes)

```bash
./test/api/kleidicv-api-test
```

Expected output:
```
[==========] 4543 tests from N test suites ran.
[  PASSED  ] 4526 tests.
[  SKIPPED ] 17 tests.
```

The 17 skipped tests are intentionally long-running. They can be enabled:

### Full test suite including long-running tests (~5 minutes)

```bash
./test/api/kleidicv-api-test --long-running-tests
```

Expected output:
```
[==========] 4543 tests from N test suites ran.
[  PASSED  ] 4543 tests.
```

The `Exp/0.AllValues` test takes ~200 seconds — it exhaustively checks all 2^32 float bit patterns.

### Run a specific test suite

```bash
./test/api/kleidicv-api-test --gtest_filter='SaturatingAdd*'
./test/api/kleidicv-api-test --gtest_filter='Yuv444*:RgbToYuv444*'
./test/api/kleidicv-api-test --gtest_filter='Sobel*:Scharr*'
```

### Brief output (failures only)

```bash
./test/api/kleidicv-api-test --gtest_brief=1
```

### Run per-suite to isolate crashes

If a test crashes and prevents later tests from running:

```bash
for suite in $(./test/api/kleidicv-api-test --gtest_list_tests | grep -E "^[A-Z]" | sed "s/ .*//" | sort -u); do
  timeout 60 ./test/api/kleidicv-api-test --gtest_brief=1 --gtest_filter="${suite}*" 2>&1 | tail -3
done
```

### Verify both RVV and scalar paths

Both builds should produce identical test results:

```bash
# RVV build
cd build && ./test/api/kleidicv-api-test --gtest_brief=1 | tail -3

# Scalar build
cd ../build_scalar && ./test/api/kleidicv-api-test --gtest_brief=1 | tail -3
```

Expected: both show `4526 passed, 0 failed, 17 skipped`.

## Benchmark

Performance comparison between RVV and scalar builds uses `scripts/bench.cpp`.

### Run

```bash
cd /path/to/rvspoc-P2601-kleidicv
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

### Verify that RVV instructions are present

```bash
# RVV build: should be >0
objdump -d build/kleidicv/libkleidicv.a | grep -c 'vsetvli'

# Scalar build: should be 0
objdump -d build_scalar/kleidicv/libkleidicv.a | grep -c 'vsetvli'
```

## Test Coverage

### Test suites by operator category

| Category | Test Suites | Tests |
|----------|------------|-------|
| Arithmetic (add, sub, absdiff, mul, ...) | SaturatingAdd, SaturatingSub, SaturatingAbsDiff, SaturatingMultiply, BitwiseAnd, Compare, ThresholdBinary, InRange, ScaleTest, Exp | ~550 |
| Conversions (gray, rgb, float, split, merge) | GRAY2, RGB2, RGBA2, FloatConversion, Split, Merge | ~400 |
| YUV conversions | YUV420p2RGBTest, YUV422i2RGBTest, RGB2YUV420pTest, etc. | ~700 |
| Analysis (sum, count, minmax) | Sum, CountNonZeros, MinMax, MinMaxLoc | ~150 |
| Filters (blur, sobel, scharr, median) | GaussianBlur, Sobel, ScharrInterleaved, MedianBlurTest, SeparableFilter2D, BlurAndDownsample | ~250 |
| Morphology (dilate, erode) | Morphology | ~30 |
| Transform (transpose, rotate, remap, warp) | Transpose, Rotate, RemapF32, RemapS16, RemapS16Point5, WarpPerspective | ~600 |
| Resize (linear, quarter) | ResizeLinear, ResizeToQuarter | ~400 |
| Optical flow | BuildOpticalFlowPyrLkPyramid, CalcOpticalFlowPyrLk, StandaloneLKAlgTest | ~200 |
| Threading | Thread*, *Thread | ~250 |

### What the tests verify

- **Correctness:** Output compared element-by-element against a reference implementation. Integer types use exact match; float types use `max_relative_error` or `nextafterf` (1-ULP) tolerance.
- **Edge cases:** NaN, infinity, subnormals, zero, min/max type values, single-pixel images, max-size images.
- **Border handling:** REPLICATE, REFLECT, WRAP, CONSTANT border types for filters.
- **Allocation failures:** `MockMallocToFail` injects `malloc` returning nullptr to verify graceful error handling.
- **Null pointers:** Passing nullptr for src/dst verifies `KLEIDICV_ERROR_NULL_POINTER`.
- **Alignment:** Misaligned strides verify `KLEIDICV_ERROR_ALIGNMENT`.
- **Image size limits:** Oversized images verify `KLEIDICV_ERROR_RANGE`.
- **Threading:** Multi-threaded dispatch correctness verified against single-threaded reference.
- **In-place operation:** Verifying src==dst works correctly where supported.

## Verify VLEN

To confirm the hardware VLEN:

```bash
cat > /tmp/vlen.c << 'EOF'
#include <stdio.h>
#include <riscv_vector.h>
int main() {
    size_t vl = __riscv_vsetvl_e8m1(65536);
    printf("VLEN = %zu bits (%zu bytes)\n", vl * 8, vl);
    return 0;
}
EOF
cc -march=rv64gcv -o /tmp/vlen /tmp/vlen.c && /tmp/vlen
```

Expected: `VLEN = 256 bits (32 bytes)`

All RVV code is VLEN-agnostic. The same binary works on VLEN=128, 256, and 512 without recompilation. The `vsetvl` instruction returns the actual hardware vector length at runtime.

## Troubleshooting

### "Text file busy" when running tests after build

The old test binary is still in use by a previous process. Kill it first:
```bash
killall kleidicv-api-test 2>/dev/null; sleep 1
```

### "Permission denied" running test binary

```bash
chmod +x test/api/kleidicv-api-test
```

### FetchContent download fails (no internet)

Use `FETCHCONTENT_SOURCE_DIR_GOOGLETEST=/tmp/gtest` as shown above.

### Test reports "Vector length is set to 16 bytes"

This is the test framework's default (Neon 128-bit = 16 bytes). It does not affect actual VLEN — the RVV code always uses `vsetvl` to query the real hardware VLEN.
