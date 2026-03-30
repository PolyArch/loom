#include "jacobi_ref.h"

#include "test_scicomp_utils.h"

#include <cmath>

namespace scicomp_test {

namespace {

static void copy_row_halo(std::vector<float> &grid, int rows, int cols,
                          int halo_w) {
  const int total_rows = rows + 2 * halo_w;
  const int total_cols = cols + 2 * halo_w;
  for (int h = 0; h < halo_w; ++h) {
    for (int col = 0; col < total_cols; ++col) {
      grid[static_cast<size_t>(h) * static_cast<size_t>(total_cols) +
           static_cast<size_t>(col)] =
          grid[static_cast<size_t>(halo_w) * static_cast<size_t>(total_cols) +
               static_cast<size_t>(col)];
      grid[static_cast<size_t>(total_rows - 1 - h) *
               static_cast<size_t>(total_cols) +
           static_cast<size_t>(col)] =
          grid[static_cast<size_t>(total_rows - 1 - halo_w) *
               static_cast<size_t>(total_cols) +
               static_cast<size_t>(col)];
    }
  }
}

static void apply_dirichlet_boundary(std::vector<float> &grid, int rows,
                                     int cols, int halo_w, float value) {
  const int total_rows = rows + 2 * halo_w;
  const int total_cols = cols + 2 * halo_w;
  for (int row = 0; row < total_rows; ++row) {
    for (int h = 0; h < halo_w; ++h) {
      grid[static_cast<size_t>(row) * static_cast<size_t>(total_cols) +
           static_cast<size_t>(h)] = value;
      grid[static_cast<size_t>(row) * static_cast<size_t>(total_cols) +
           static_cast<size_t>(total_cols - 1 - h)] = value;
    }
  }
  for (int col = 0; col < total_cols; ++col) {
    for (int h = 0; h < halo_w; ++h) {
      grid[static_cast<size_t>(h) * static_cast<size_t>(total_cols) +
           static_cast<size_t>(col)] = value;
      grid[static_cast<size_t>(total_rows - 1 - h) *
               static_cast<size_t>(total_cols) +
           static_cast<size_t>(col)] = value;
    }
  }
}

} // namespace

JacobiReferenceResult run_jacobi_reference(const std::vector<float> &initial,
                                           int rows, int cols, int halo_w,
                                           float factor, int max_iters,
                                           float stop_eps) {
  const int total_rows = rows + 2 * halo_w;
  const int total_cols = cols + 2 * halo_w;
  JacobiReferenceResult result;
  result.grid = initial;

  std::vector<float> next = initial;

  for (int iter = 0; iter < max_iters; ++iter) {
    copy_row_halo(result.grid, rows, cols, halo_w);

    for (int row = halo_w; row < rows + halo_w; ++row) {
      for (int col = halo_w; col < cols + halo_w; ++col) {
        const size_t idx = static_cast<size_t>(row) *
                               static_cast<size_t>(total_cols) +
                           static_cast<size_t>(col);
        const float center = result.grid[idx];
        const float sum = result.grid[idx - static_cast<size_t>(total_cols)] +
                          result.grid[idx + static_cast<size_t>(total_cols)] +
                          result.grid[idx - 1] + result.grid[idx + 1] -
                          4.0f * center;
        next[idx] = sum * factor;
      }
    }

    apply_dirichlet_boundary(next, rows, cols, halo_w, 0.0f);
    float sum_sq = 0.0f;
    for (int row = 0; row < total_rows; ++row) {
      for (int col = 0; col < total_cols; ++col) {
        const size_t idx = static_cast<size_t>(row) *
                               static_cast<size_t>(total_cols) +
                           static_cast<size_t>(col);
        const float diff = next[idx] - result.grid[idx];
        sum_sq += diff * diff;
      }
    }
    result.grid.swap(next);
    result.residual = std::sqrt(sum_sq);
    result.iterations = iter + 1;
    if (result.residual < stop_eps)
      break;
  }

  return result;
}

} // namespace scicomp_test
