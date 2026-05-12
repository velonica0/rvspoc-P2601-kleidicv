// SPDX-FileCopyrightText: 2026 RVSPOC Contributors
//
// SPDX-License-Identifier: Apache-2.0

#include "kleidicv/analysis/standalone_lucas_kanade_alg.h"
#include "kleidicv/rvv.h"
#include "standalone_lucas_kanade_alg_common.h"

namespace kleidicv::neon {

// Scalar fallback -- the NEON LK implementation uses heavy NEON-specific
// intrinsics (interleaved loads, widening multiply-accumulate, etc.) that
// require a dedicated RVV rewrite.  Delegate to the scalar implementation
// from the _sc.h header which the upstream codebase already provides.

class StandaloneLucasKanadeAlgScalar {
 public:
  KLEIDICV_FORCE_INLINE
  static void sample_patch_and_gradients(
      int16_t *window, int16_t *scharr_window, const uint8_t *prev_data,
      ptrdiff_t prev_data_stride, const int16_t *scharr_data,
      ptrdiff_t scharr_stride_elements, int channels, int window_corner_x,
      int window_corner_y, int window_width, int window_height,
      int16_t coeff_tl, int16_t coeff_tr, int16_t coeff_bl, int16_t coeff_br,
      float &sum_scharr_xx, float &sum_scharr_xy, float &sum_scharr_yy) {
    sum_scharr_xx = 0;
    sum_scharr_xy = 0;
    sum_scharr_yy = 0;

    const int window_width_times_channels = window_width * channels;

    for (int y = 0; y < window_height; y++) {
      const uint8_t *const prev_row0 =
          prev_data + (y + window_corner_y) * prev_data_stride +
          static_cast<ptrdiff_t>(window_corner_x) * channels;
      const uint8_t *const prev_row1 = prev_row0 + prev_data_stride;
      int16_t *const window_row =
          window + static_cast<ptrdiff_t>(y) * window_width * channels;

      const int16_t *const scharr_row0 =
          scharr_data + (y + window_corner_y) * scharr_stride_elements +
          static_cast<ptrdiff_t>(window_corner_x) * channels * 2L;
      const int16_t *const scharr_row1 = scharr_row0 + scharr_stride_elements;
      int16_t *const scharr_window_row =
          scharr_window +
          static_cast<ptrdiff_t>(y) * window_width * channels * 2L;

      for (int x = 0; x < window_width_times_channels; ++x) {
        window_row[x] = static_cast<int16_t>(
            round_fixed_point<kLucasKanadeAlgFractionBits - 5>(
                prev_row0[x] * coeff_tl +
                prev_row0[x + channels] * coeff_tr + prev_row1[x] * coeff_bl +
                prev_row1[x + channels] * coeff_br));

        int scharr_x = round_fixed_point<kLucasKanadeAlgFractionBits>(
            scharr_row0[x * 2L] * coeff_tl +
            scharr_row0[(x + channels) * 2L] * coeff_tr +
            scharr_row1[x * 2L] * coeff_bl +
            scharr_row1[(x + channels) * 2L] * coeff_br);
        int scharr_y = round_fixed_point<kLucasKanadeAlgFractionBits>(
            scharr_row0[x * 2L + 1] * coeff_tl +
            scharr_row0[(x + channels) * 2L + 1] * coeff_tr +
            scharr_row1[x * 2L + 1] * coeff_bl +
            scharr_row1[(x + channels) * 2L + 1] * coeff_br);
        scharr_window_row[x * 2L] = static_cast<int16_t>(scharr_x);
        scharr_window_row[x * 2L + 1] = static_cast<int16_t>(scharr_y);
        sum_scharr_xx += static_cast<float>(scharr_x * scharr_x);
        sum_scharr_xy += static_cast<float>(scharr_x * scharr_y);
        sum_scharr_yy += static_cast<float>(scharr_y * scharr_y);
      }
    }
    sum_scharr_xx *= kLucasKanadeAlgFixedPointDescale;
    sum_scharr_xy *= kLucasKanadeAlgFixedPointDescale;
    sum_scharr_yy *= kLucasKanadeAlgFixedPointDescale;
  }

  KLEIDICV_FORCE_INLINE
  static void accumulate_mismatch_vector(
      const uint8_t *next_data, ptrdiff_t next_stride, const int16_t *window,
      const int16_t *scharr_window, int channels, int window_corner_x,
      int window_corner_y, int window_width, int window_height,
      int16_t coeff_tl, int16_t coeff_tr, int16_t coeff_bl, int16_t coeff_br,
      float &sum_diff_scharr_x, float &sum_diff_scharr_y) {
    sum_diff_scharr_x = 0;
    sum_diff_scharr_y = 0;

    const int window_width_times_channels = window_width * channels;

    for (int y = 0; y < window_height; y++) {
      const uint8_t *row0 = next_data + (y + window_corner_y) * next_stride +
                            static_cast<ptrdiff_t>(window_corner_x) * channels;
      const uint8_t *row1 = row0 + next_stride;
      const int16_t *window_row =
          window + static_cast<ptrdiff_t>(y) * window_width * channels;
      const int16_t *scharr_window_row =
          scharr_window +
          static_cast<ptrdiff_t>(y) * window_width * channels * 2L;

      for (int x = 0; x < window_width_times_channels; ++x) {
        int next = round_fixed_point<kLucasKanadeAlgFractionBits - 5>(
            row0[x] * coeff_tl + row0[x + channels] * coeff_tr +
            row1[x] * coeff_bl + row1[x + channels] * coeff_br);
        int diff = next - window_row[x];
        sum_diff_scharr_x +=
            static_cast<float>(diff * scharr_window_row[x * 2L]);
        sum_diff_scharr_y +=
            static_cast<float>(diff * scharr_window_row[x * 2L + 1]);
      }
    }
    sum_diff_scharr_x *= kLucasKanadeAlgFixedPointDescale;
    sum_diff_scharr_y *= kLucasKanadeAlgFixedPointDescale;
  }
};

KLEIDICV_TARGET_FN_ATTRS kleidicv_error_t standalone_lucas_kanade_alg_u8(
    const uint8_t *prev_data, size_t prev_data_stride,
    const int16_t *prev_deriv_data, size_t prev_deriv_stride,
    const uint8_t *next_data, size_t next_data_stride, int width, int height,
    int channels, const float *prev_points, float *next_points,
    size_t point_count, uint8_t *status, float *err, int window_width,
    int window_height, int termination_count, double termination_epsilon,
    bool get_min_eigen_vals, float min_eigen_vals_threshold) {
  if (kleidicv_error_t validation_error =
          validate_standalone_lucas_kanade_alg_u8_args(
              prev_data, prev_data_stride, prev_deriv_data, prev_deriv_stride,
              next_data, next_data_stride, width, height, channels, prev_points,
              next_points, point_count, window_width, window_height)) {
    return validation_error;
  }

  auto window_buffer_or_error =
      LucasKanadePatchBuffer<>::create(window_width, window_height, channels);
  if (!std::holds_alternative<LucasKanadePatchBuffer<>>(
          window_buffer_or_error)) {
    return std::get<kleidicv_error_t>(window_buffer_or_error);
  }
  auto &window_buffer =
      std::get<LucasKanadePatchBuffer<>>(window_buffer_or_error);
  return LucasKanadeLevelTracker<StandaloneLucasKanadeAlgScalar>::compute(
      window_buffer.window(), window_buffer.deriv_window(), prev_data,
      prev_data_stride, prev_deriv_data, prev_deriv_stride, next_data,
      next_data_stride, width, height, channels, prev_points, next_points,
      point_count, status, err, window_width, window_height, termination_count,
      termination_epsilon, get_min_eigen_vals, min_eigen_vals_threshold);
}

}  // namespace kleidicv::neon
