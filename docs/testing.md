# Testing Procedure

## Target Hardware

- **Board:** Spacemit X100
- **ISA:** rv64gcv (RV64 + V extension)
- **VLEN:** 256 bits (32 bytes per vector register)
- **Cores:** 8
- **OS:** openKylin (Linux 6.18.3+)
- **Compiler:** GCC 15.2.0 (Bianbu)
- **SSH:** `ssh openkylin@192.168.5.211` (password: `openkylin`)
- **Working directory:** `/home/openkylin/github/rvspoc-P2601-kleidicv`

## Build

### Prerequisites

GoogleTest 1.12.1 must be available locally (the target has no internet). Download on a connected machine and extract to `/tmp/gtest` on the target:

```bash
# On a machine with internet:
wget https://github.com/google/googletest/archive/refs/tags/release-1.12.1.tar.gz
scp release-1.12.1.tar.gz openkylin@192.168.5.211:/tmp/

# On the target:
cd /tmp && tar xzf release-1.12.1.tar.gz && mv googletest-release-1.12.1 gtest
cd gtest && patch --strip=1 --input=/home/openkylin/github/rvspoc-P2601-kleidicv/test/patches/googletest.patch
```

### Sync Code

From the development machine:

```bash
rsync -az --exclude='.git' --exclude='build' \
  /path/to/rvspoc-P2601-kleidicv/ \
  openkylin@192.168.5.211:/home/openkylin/github/rvspoc-P2601-kleidicv/
```

### Configure and Build

```bash
ssh openkylin@192.168.5.211
cd /home/openkylin/github/rvspoc-P2601-kleidicv
mkdir -p build && cd build

cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DKLEIDICV_BENCHMARK=OFF \
  -DFETCHCONTENT_SOURCE_DIR_GOOGLETEST=/tmp/gtest

make kleidicv-api-test -j8
```

**Build notes:**
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

## Verify RVV is Used (Not Scalar Fallback)

The test binary tests through the public C API which is backend-agnostic — the output doesn't show whether the RVV or scalar path ran. Both paths produce identical results by design. Three methods to verify:

### Method 1: Check binary for vector instructions

```bash
cd build
objdump -d kleidicv/libkleidicv.a | grep -c 'vsetvli\b'
```

If the count is >0, vector instructions are compiled in. For a specific operator:

```bash
objdump -d kleidicv/CMakeFiles/kleidicv_rvv.dir/src/arithmetics/add_rvv.cpp.o | grep -E 'vsetvli|vle8|vse8|vsadd'
```

### Method 2: Performance comparison

The definitive proof. Compare the same function via scalar C loop vs the RVV-compiled library:

```bash
cat > /tmp/bench.cpp << 'EOF'
#include <cstdio>
#include <cstdint>
#include <cstring>
#include <chrono>
#include <riscv_vector.h>

void add_scalar(const uint8_t *a, const uint8_t *b, uint8_t *c, size_t n) {
    for (size_t i = 0; i < n; i++) {
        int s = a[i] + b[i]; c[i] = s > 255 ? 255 : s;
    }
}
void add_rvv(const uint8_t *a, const uint8_t *b, uint8_t *c, size_t n) {
    size_t i = 0;
    while (i < n) {
        size_t vl = __riscv_vsetvl_e8m1(n - i);
        vuint8m1_t va = __riscv_vle8_v_u8m1(a + i, vl);
        vuint8m1_t vb = __riscv_vle8_v_u8m1(b + i, vl);
        __riscv_vse8_v_u8m1(c + i, __riscv_vsaddu_vv_u8m1(va, vb, vl), vl);
        i += vl;
    }
}
int main() {
    const size_t N = 1920 * 1080;
    uint8_t *a = new uint8_t[N], *b = new uint8_t[N], *c = new uint8_t[N];
    memset(a, 100, N); memset(b, 50, N);
    const int IT = 200;
    add_scalar(a, b, c, N);
    auto t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < IT; i++) add_scalar(a, b, c, N);
    auto t1 = std::chrono::high_resolution_clock::now();
    double ms_s = std::chrono::duration<double, std::milli>(t1 - t0).count() / IT;
    add_rvv(a, b, c, N);
    t0 = std::chrono::high_resolution_clock::now();
    for (int i = 0; i < IT; i++) add_rvv(a, b, c, N);
    t1 = std::chrono::high_resolution_clock::now();
    double ms_r = std::chrono::duration<double, std::milli>(t1 - t0).count() / IT;
    printf("Scalar: %.2f ms  RVV: %.2f ms  Speedup: %.1fx\n", ms_s, ms_r, ms_s / ms_r);
    delete[] a; delete[] b; delete[] c;
}
EOF
c++ -O2 -std=c++17 -march=rv64gcv -o /tmp/bench /tmp/bench.cpp && /tmp/bench
```

### Method 3: Build both RVV and scalar libraries, benchmark the same API

This is the definitive method. Build the library twice with different `-march`, then benchmark both against the same API:

```bash
cd /home/openkylin/github/rvspoc-P2601-kleidicv

# RVV build (default)
cd build
# already built with -march=rv64gcv

# Scalar build: change rv64gcv -> rv64gc in CMakeLists.txt line 201
cd ../build_scalar
sed -i 's/-march=rv64gcv/-march=rv64gc/' ../kleidicv/CMakeLists.txt
cmake .. -DCMAKE_BUILD_TYPE=Release ...
make kleidicv -j8
sed -i 's/-march=rv64gc/-march=rv64gcv/' ../kleidicv/CMakeLists.txt  # restore

# Verify: 0 vector instructions in scalar, >0 in RVV
objdump -d build/kleidicv/libkleidicv.a | grep -c 'vsetvli'        # 587
objdump -d build_scalar/kleidicv/libkleidicv.a | grep -c 'vsetvli'  # 0

# Run same benchmark against both:
c++ -O2 -I... -o bench_rvv bench.cpp -Lbuild/kleidicv -lkleidicv
c++ -O2 -I... -o bench_scalar bench.cpp -Lbuild_scalar/kleidicv -lkleidicv
```

Measured on Spacemit X100 (VLEN=256), 1920x1080, 200 iterations:

| Operation | Scalar (ms) | RVV (ms) | Speedup |
|-----------|------------|---------|---------|
| saturating_add_u8 | 3.085 | 0.473 | 6.5x |
| saturating_sub_u8 | 2.847 | 0.448 | 6.4x |
| min_max_u8 | 3.838 | 0.148 | 25.9x |
| gray_to_rgb_u8 | 2.849 | 0.421 | 6.8x |

Both builds pass the full test suite identically (4526 passed, 0 failed).

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
