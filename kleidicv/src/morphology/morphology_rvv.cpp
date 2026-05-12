// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include <algorithm>
#include <cstring>
#include <limits>

#include "kleidicv/morphology/morphology.h"
#include "kleidicv/morphology/workspace.h"
#include "kleidicv/rvv.h"

namespace kleidicv::neon {

// ---------------------------------------------------------------------------
// Scalar vertical operations for morphology (min / max across kernel rows)
// ---------------------------------------------------------------------------
template <typename ScalarType, typename Op>
class RvvVerticalOp final {
 public:
  RvvVerticalOp(Rectangle rect, Rectangle kernel)
      : rect_(rect), kernel_(kernel) {}

  void process_rows(IndirectRows<ScalarType> src_rows,
                    Rows<ScalarType> dst_rows) {
    if (KLEIDICV_UNLIKELY(kernel_.height() == 1)) {
      CopyRows<ScalarType>::copy_rows(rect_, src_rows, dst_rows);
      return;
    }

    const size_t total_width = rect_.width() * src_rows.channels();

    for (size_t h = 0; h < rect_.height(); ++h) {
      // First kernel row initializes the accumulator.
      const ScalarType *row0 = &src_rows.at(0)[0];

      size_t x = 0;

#ifdef __riscv_vector
      // RVV vectorized path for the row.
      while (x < total_width) {
        size_t vl = rvv_setvl<ScalarType>(total_width - x);

        auto acc = rvv_load(row0 + x, vl);

        for (size_t ky = 1; ky < kernel_.height(); ++ky) {
          const ScalarType *row_k = &src_rows.at(static_cast<ptrdiff_t>(ky))[0];
          auto v = rvv_load(row_k + x, vl);
          acc = Op::rvv_op(acc, v, vl);
        }

        rvv_store(&dst_rows[static_cast<ptrdiff_t>(x)], acc, vl);
        x += vl;
      }
#else
      // Pure scalar path (no RVV).
      for (; x < total_width; ++x) {
        ScalarType acc = row0[x];
        for (size_t ky = 1; ky < kernel_.height(); ++ky) {
          const ScalarType *row_k = &src_rows.at(static_cast<ptrdiff_t>(ky))[0];
          acc = Op::scalar_op(acc, row_k[x]);
        }
        dst_rows[static_cast<ptrdiff_t>(x)] = acc;
      }
#endif

      ++src_rows;
      ++dst_rows;
    }
  }

 private:
  Rectangle rect_;
  Rectangle kernel_;
};

// ---------------------------------------------------------------------------
// Scalar horizontal operations for morphology (min / max across kernel cols)
// ---------------------------------------------------------------------------
template <typename ScalarType, typename Op>
class RvvHorizontalOp final {
 public:
  RvvHorizontalOp(Rectangle rect, Rectangle kernel)
      : rect_(rect), kernel_(kernel) {}

  void process_rows(Rows<const ScalarType> src_rows,
                    Rows<ScalarType> dst_rows) {
    const size_t channels = src_rows.channels();

    for (size_t h = 0; h < rect_.height(); ++h) {
      const size_t total_width = rect_.width() * channels;

      size_t x = 0;

#ifdef __riscv_vector
      // RVV vectorized horizontal path.
      while (x < total_width) {
        size_t vl = rvv_setvl<ScalarType>(total_width - x);

        auto acc = rvv_load(&src_rows[static_cast<ptrdiff_t>(x)], vl);
        for (size_t kw = 1; kw < kernel_.width(); ++kw) {
          ptrdiff_t offset = static_cast<ptrdiff_t>(x + kw * channels);
          auto v = rvv_load(&src_rows[offset], vl);
          acc = Op::rvv_op(acc, v, vl);
        }

        rvv_store(&dst_rows[static_cast<ptrdiff_t>(x)], acc, vl);
        x += vl;
      }
#else
      // Pure scalar horizontal path.
      for (; x < total_width; ++x) {
        ScalarType acc = src_rows[static_cast<ptrdiff_t>(x)];
        for (size_t kw = 1; kw < kernel_.width(); ++kw) {
          ptrdiff_t offset = static_cast<ptrdiff_t>(x + kw * channels);
          acc = Op::scalar_op(acc, src_rows[offset]);
        }
        dst_rows[static_cast<ptrdiff_t>(x)] = acc;
      }
#endif

      ++src_rows;
      ++dst_rows;
    }
  }

 private:
  Rectangle rect_;
  Rectangle kernel_;
};

// ---------------------------------------------------------------------------
// RVV vector type trait: maps scalar type to its m1 vector type.
// ---------------------------------------------------------------------------
#ifdef __riscv_vector
template <typename T>
struct RvvVecType;

template <>
struct RvvVecType<uint8_t> {
  using type = vuint8m1_t;
};
#endif

// ---------------------------------------------------------------------------
// Operation traits: min (erode) and max (dilate)
// ---------------------------------------------------------------------------
template <typename ScalarType>
struct MinOp {
  static ScalarType scalar_op(ScalarType a, ScalarType b) {
    return std::min(a, b);
  }
#ifdef __riscv_vector
  using VecType = typename RvvVecType<ScalarType>::type;
  static VecType rvv_op(VecType a, VecType b, size_t vl) {
    return rvv_min(a, b, vl);
  }
#endif
};

template <typename ScalarType>
struct MaxOp {
  static ScalarType scalar_op(ScalarType a, ScalarType b) {
    return std::max(a, b);
  }
#ifdef __riscv_vector
  using VecType = typename RvvVecType<ScalarType>::type;
  static VecType rvv_op(VecType a, VecType b, size_t vl) {
    return rvv_max(a, b, vl);
  }
#endif
};

// ---------------------------------------------------------------------------
// DilateOperation / ErodeOperation wrappers for MorphologyWorkspace::process()
// ---------------------------------------------------------------------------
template <typename ScalarType>
class RvvDilateOperation final {
 public:
  using SourceType = ScalarType;
  using BufferType = ScalarType;
  using DestinationType = ScalarType;
  using CopyData = MorphologyWorkspace::CopyDataMemcpy<ScalarType>;

  explicit RvvDilateOperation(Rectangle kernel) : kernel_{kernel} {}

  void process_horizontal(Rectangle rect, Rows<const SourceType> src_rows,
                          Rows<BufferType> dst_rows) {
    RvvHorizontalOp<ScalarType, MaxOp<ScalarType>>{rect, kernel_}.process_rows(
        src_rows, dst_rows);
  }

  void process_vertical(Rectangle rect, IndirectRows<BufferType> src_rows,
                        Rows<DestinationType> dst_rows) {
    RvvVerticalOp<ScalarType, MaxOp<ScalarType>>{rect, kernel_}.process_rows(
        src_rows, dst_rows);
  }

 private:
  Rectangle kernel_;
};

template <typename ScalarType>
class RvvErodeOperation final {
 public:
  using SourceType = ScalarType;
  using BufferType = ScalarType;
  using DestinationType = ScalarType;
  using CopyData = MorphologyWorkspace::CopyDataMemcpy<ScalarType>;

  explicit RvvErodeOperation(Rectangle kernel) : kernel_{kernel} {}

  void process_horizontal(Rectangle rect, Rows<const SourceType> src_rows,
                          Rows<BufferType> dst_rows) {
    RvvHorizontalOp<ScalarType, MinOp<ScalarType>>{rect, kernel_}.process_rows(
        src_rows, dst_rows);
  }

  void process_vertical(Rectangle rect, IndirectRows<BufferType> src_rows,
                        Rows<DestinationType> dst_rows) {
    RvvVerticalOp<ScalarType, MinOp<ScalarType>>{rect, kernel_}.process_rows(
        src_rows, dst_rows);
  }

 private:
  Rectangle kernel_;
};

// ---------------------------------------------------------------------------
// dilate (max)
// ---------------------------------------------------------------------------
template <typename T>
kleidicv_error_t dilate(const T *src, size_t src_stride, T *dst,
                        size_t dst_stride, size_t width, size_t height,
                        size_t channels, size_t kernel_width,
                        size_t kernel_height, size_t anchor_x, size_t anchor_y,
                        kleidicv_border_type_t border_type,
                        const uint8_t *border_value, size_t iterations) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);
  CHECK_IMAGE_SIZE(kernel_width, kernel_height);

  auto morphology_border_type =
      MorphologyWorkspace::get_border_type(border_type);
  if (!morphology_border_type) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
  if (!morphology_is_implemented(width, height, kernel_width, kernel_height,
                                 channels)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  Rectangle rect{width, height};
  Rectangle kernel_rect{kernel_width, kernel_height};
  Point anchor{anchor_x, anchor_y};

  auto workspace_variant = MorphologyWorkspace::create(
      kernel_rect, anchor, *morphology_border_type, border_value, channels,
      sizeof(T), rect);
  if (auto *err = std::get_if<kleidicv_error_t>(&workspace_variant)) {
    return *err;
  }
  auto &workspace = *std::get_if<MorphologyWorkspace>(&workspace_variant);

  Rows<const T> src_rows{src, src_stride, channels};
  Rows<T> dst_rows{dst, dst_stride, channels};

  Rows<const T> current_src_rows = src_rows;
  Rows<T> current_dst_rows = dst_rows;
  for (size_t i = 0; i < iterations; ++i) {
    RvvDilateOperation<T> operation{kernel_rect};
    workspace.process(current_src_rows, current_dst_rows, operation);
    // Update source for the next iteration.
    current_src_rows = dst_rows;
  }
  return KLEIDICV_OK;
}

// ---------------------------------------------------------------------------
// erode (min)
// ---------------------------------------------------------------------------
template <typename T>
kleidicv_error_t erode(const T *src, size_t src_stride, T *dst,
                       size_t dst_stride, size_t width, size_t height,
                       size_t channels, size_t kernel_width,
                       size_t kernel_height, size_t anchor_x, size_t anchor_y,
                       kleidicv_border_type_t border_type,
                       const uint8_t *border_value, size_t iterations) {
  CHECK_POINTER_AND_STRIDE(src, src_stride, height);
  CHECK_POINTER_AND_STRIDE(dst, dst_stride, height);
  CHECK_IMAGE_SIZE(width, height);
  CHECK_IMAGE_SIZE(kernel_width, kernel_height);

  auto morphology_border_type =
      MorphologyWorkspace::get_border_type(border_type);
  if (!morphology_border_type) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }
  if (!morphology_is_implemented(width, height, kernel_width, kernel_height,
                                 channels)) {
    return KLEIDICV_ERROR_NOT_IMPLEMENTED;
  }

  Rectangle rect{width, height};
  Rectangle kernel_rect{kernel_width, kernel_height};
  Point anchor{anchor_x, anchor_y};

  auto workspace_variant = MorphologyWorkspace::create(
      kernel_rect, anchor, *morphology_border_type, border_value, channels,
      sizeof(T), rect);
  if (auto *err = std::get_if<kleidicv_error_t>(&workspace_variant)) {
    return *err;
  }
  auto &workspace = *std::get_if<MorphologyWorkspace>(&workspace_variant);

  Rows<const T> src_rows{src, src_stride, channels};
  Rows<T> dst_rows{dst, dst_stride, channels};

  Rows<const T> current_src_rows = src_rows;
  Rows<T> current_dst_rows = dst_rows;
  for (size_t i = 0; i < iterations; ++i) {
    RvvErodeOperation<T> operation{kernel_rect};
    workspace.process(current_src_rows, current_dst_rows, operation);
    // Update source for the next iteration.
    current_src_rows = dst_rows;
  }
  return KLEIDICV_OK;
}

#define KLEIDICV_INSTANTIATE_TEMPLATE(name, type)                        \
  template KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t name<type>(         \
      const type *src, size_t src_stride, type *dst, size_t dst_stride,  \
      size_t width, size_t height, size_t channels, size_t kernel_width, \
      size_t kernel_height, size_t anchor_x, size_t anchor_y,            \
      kleidicv_border_type_t border_type, const uint8_t *border_value,   \
      size_t iterations)

KLEIDICV_INSTANTIATE_TEMPLATE(dilate, uint8_t);
KLEIDICV_INSTANTIATE_TEMPLATE(erode, uint8_t);

}  // namespace kleidicv::neon
