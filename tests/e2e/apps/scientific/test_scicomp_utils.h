#ifndef TAPESTRY_TEST_SCICOMP_UTILS_H
#define TAPESTRY_TEST_SCICOMP_UTILS_H

#include "kernels/scicomp_types.h"

#include <algorithm>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <vector>

namespace scicomp_test {

inline bool nearly_equal(float lhs, float rhs, float abs_eps, float rel_eps) {
  const float diff = std::fabs(lhs - rhs);
  const float scale = std::max(std::fabs(lhs), std::fabs(rhs));
  return diff <= std::max(abs_eps, rel_eps * scale);
}

inline bool nearly_equal(float lhs, float rhs, float eps = 1e-4f) {
  return nearly_equal(lhs, rhs, eps, eps);
}

inline std::vector<float> make_jacobi_grid(int rows, int cols, int halo_w) {
  const int total_rows = rows + 2 * halo_w;
  const int total_cols = cols + 2 * halo_w;
  return std::vector<float>(static_cast<size_t>(total_rows) *
                                static_cast<size_t>(total_cols),
                            0.0f);
}

inline void init_jacobi_problem(std::vector<float> &grid, int rows, int cols,
                                int halo_w) {
  const int total_cols = cols + 2 * halo_w;
  const int total_rows = rows + 2 * halo_w;
  for (int row = halo_w; row < rows + halo_w; ++row) {
    for (int col = halo_w; col < cols + halo_w; ++col) {
      const float x = static_cast<float>(row - halo_w + 1);
      const float y = static_cast<float>(col - halo_w + 1);
      grid[static_cast<size_t>(row) * static_cast<size_t>(total_cols) +
           static_cast<size_t>(col)] = std::sin(0.25f * x) *
                                       std::cos(0.5f * y);
    }
  }
  for (int row = 0; row < total_rows; ++row) {
    for (int h = 0; h < halo_w; ++h) {
      grid[static_cast<size_t>(row) * static_cast<size_t>(total_cols) +
           static_cast<size_t>(h)] = 0.0f;
      grid[static_cast<size_t>(row) * static_cast<size_t>(total_cols) +
           static_cast<size_t>(total_cols - 1 - h)] = 0.0f;
    }
  }
  for (int col = 0; col < total_cols; ++col) {
    for (int h = 0; h < halo_w; ++h) {
      grid[static_cast<size_t>(h) * static_cast<size_t>(total_cols) +
           static_cast<size_t>(col)] = 0.0f;
      grid[static_cast<size_t>(total_rows - 1 - h) *
           static_cast<size_t>(total_cols) +
           static_cast<size_t>(col)] = 0.0f;
    }
  }
}

inline bool grids_close(const std::vector<float> &lhs,
                        const std::vector<float> &rhs, float eps = 1e-4f) {
  if (lhs.size() != rhs.size())
    return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!nearly_equal(lhs[i], rhs[i], eps))
      return false;
  }
  return true;
}

struct CsrMatrixF32 {
  int rows = 0;
  int cols = 0;
  std::vector<int> row_ptr;
  std::vector<int> col_idx;
  std::vector<float> values;
};

struct EllMatrixF32 {
  int rows = 0;
  int cols = 0;
  int max_nnz = 0;
  std::vector<int> col_idx;
  std::vector<float> values;
};

struct NBodyState {
  std::vector<float> px;
  std::vector<float> py;
  std::vector<float> pz;
  std::vector<float> vx;
  std::vector<float> vy;
  std::vector<float> vz;
  std::vector<float> fx;
  std::vector<float> fy;
  std::vector<float> fz;
  std::vector<float> mass;

  int size() const { return static_cast<int>(px.size()); }
};

using CsrMatrixData = CsrMatrixF32;
using EllMatrixData = EllMatrixF32;
using ParticleStateData = NBodyState;

inline uint32_t next_random_u32(uint32_t &state) {
  state = state * 1664525u + 1013904223u;
  return state;
}

inline float random_unit_float(uint32_t &state) {
  const uint32_t raw = next_random_u32(state) >> 8;
  return static_cast<float>(raw & 0x00FFFFFFu) / 16777216.0f;
}

inline float random_float(uint32_t &state, float lo, float hi) {
  return lo + (hi - lo) * random_unit_float(state);
}

inline std::vector<float> make_random_vector(int n, uint32_t seed,
                                             float lo = -1.0f,
                                             float hi = 1.0f) {
  std::vector<float> values(static_cast<size_t>(n));
  uint32_t state = seed;
  for (int i = 0; i < n; ++i)
    values[static_cast<size_t>(i)] = random_float(state, lo, hi);
  return values;
}

inline bool nearly_equal_rel(float lhs, float rhs, float abs_eps = 1e-4f,
                             float rel_eps = 1e-4f) {
  return nearly_equal(lhs, rhs, abs_eps, rel_eps);
}

inline std::vector<float> csr_matvec(const CsrMatrixF32 &matrix,
                                     const std::vector<float> &x) {
  std::vector<float> y(static_cast<size_t>(matrix.rows), 0.0f);
  for (int row = 0; row < matrix.rows; ++row) {
    float sum = 0.0f;
    const int begin = matrix.row_ptr[static_cast<size_t>(row)];
    const int end = matrix.row_ptr[static_cast<size_t>(row + 1)];
    for (int idx = begin; idx < end; ++idx) {
      sum += matrix.values[static_cast<size_t>(idx)] *
             x[static_cast<size_t>(matrix.col_idx[static_cast<size_t>(idx)])];
    }
    y[static_cast<size_t>(row)] = sum;
  }
  return y;
}

inline std::vector<float> csr_diagonal(const CsrMatrixF32 &matrix) {
  std::vector<float> diag(static_cast<size_t>(matrix.rows), 0.0f);
  for (int row = 0; row < matrix.rows; ++row) {
    const int begin = matrix.row_ptr[static_cast<size_t>(row)];
    const int end = matrix.row_ptr[static_cast<size_t>(row + 1)];
    for (int idx = begin; idx < end; ++idx) {
      if (matrix.col_idx[static_cast<size_t>(idx)] == row) {
        diag[static_cast<size_t>(row)] = matrix.values[static_cast<size_t>(idx)];
        break;
      }
    }
  }
  return diag;
}

inline CsrMatrixF32 make_diagonal_csr(int n, float diag_value = 2.0f) {
  CsrMatrixF32 matrix;
  matrix.rows = n;
  matrix.cols = n;
  matrix.row_ptr.resize(static_cast<size_t>(n) + 1);
  matrix.col_idx.resize(static_cast<size_t>(n));
  matrix.values.resize(static_cast<size_t>(n));
  for (int i = 0; i < n; ++i) {
    matrix.row_ptr[static_cast<size_t>(i)] = i;
    matrix.col_idx[static_cast<size_t>(i)] = i;
    matrix.values[static_cast<size_t>(i)] = diag_value;
  }
  matrix.row_ptr[static_cast<size_t>(n)] = n;
  return matrix;
}

inline CsrMatrixF32 make_tridiagonal_laplacian_csr(int n) {
  CsrMatrixF32 matrix;
  matrix.rows = n;
  matrix.cols = n;
  matrix.row_ptr.reserve(static_cast<size_t>(n) + 1);
  matrix.row_ptr.push_back(0);
  for (int row = 0; row < n; ++row) {
    if (row > 0) {
      matrix.col_idx.push_back(row - 1);
      matrix.values.push_back(-1.0f);
    }
    matrix.col_idx.push_back(row);
    matrix.values.push_back(2.0f);
    if (row + 1 < n) {
      matrix.col_idx.push_back(row + 1);
      matrix.values.push_back(-1.0f);
    }
    matrix.row_ptr.push_back(static_cast<int>(matrix.col_idx.size()));
  }
  return matrix;
}

inline CsrMatrixF32 make_2d_laplacian_csr(int side) {
  const int n = side * side;
  CsrMatrixF32 matrix;
  matrix.rows = n;
  matrix.cols = n;
  matrix.row_ptr.reserve(static_cast<size_t>(n) + 1);
  matrix.row_ptr.push_back(0);

  for (int row = 0; row < side; ++row) {
    for (int col = 0; col < side; ++col) {
      const int idx = row * side + col;
      matrix.col_idx.push_back(idx);
      matrix.values.push_back(4.0f);
      if (row > 0) {
        matrix.col_idx.push_back(idx - side);
        matrix.values.push_back(-1.0f);
      }
      if (row + 1 < side) {
        matrix.col_idx.push_back(idx + side);
        matrix.values.push_back(-1.0f);
      }
      if (col > 0) {
        matrix.col_idx.push_back(idx - 1);
        matrix.values.push_back(-1.0f);
      }
      if (col + 1 < side) {
        matrix.col_idx.push_back(idx + 1);
        matrix.values.push_back(-1.0f);
      }
      matrix.row_ptr.push_back(static_cast<int>(matrix.col_idx.size()));
    }
  }

  return matrix;
}

inline void make_full_neighbor_list(int n, std::vector<int> &offsets,
                                    std::vector<int> &indices) {
  offsets.resize(static_cast<size_t>(n) + 1);
  indices.clear();
  indices.reserve(static_cast<size_t>(std::max(0, n * (n - 1))));
  int cursor = 0;
  for (int i = 0; i < n; ++i) {
    offsets[static_cast<size_t>(i)] = cursor;
    for (int j = 0; j < n; ++j) {
      if (i == j)
        continue;
      indices.push_back(j);
      ++cursor;
    }
  }
  offsets[static_cast<size_t>(n)] = cursor;
}

inline EllMatrixF32 csr_to_ell(const CsrMatrixF32 &csr) {
  EllMatrixF32 ell;
  ell.rows = csr.rows;
  ell.cols = csr.cols;
  int max_nnz = 0;
  for (int row = 0; row < csr.rows; ++row) {
    const int row_nnz = csr.row_ptr[static_cast<size_t>(row + 1)] -
                        csr.row_ptr[static_cast<size_t>(row)];
    max_nnz = std::max(max_nnz, row_nnz);
  }
  ell.max_nnz = max_nnz;
  ell.col_idx.assign(static_cast<size_t>(csr.rows) * static_cast<size_t>(max_nnz),
                     -1);
  ell.values.assign(static_cast<size_t>(csr.rows) * static_cast<size_t>(max_nnz),
                    0.0f);

  for (int row = 0; row < csr.rows; ++row) {
    const int row_begin = csr.row_ptr[static_cast<size_t>(row)];
    const int row_end = csr.row_ptr[static_cast<size_t>(row + 1)];
    for (int slot = 0; slot < row_end - row_begin; ++slot) {
      const int src = row_begin + slot;
      const size_t dst = static_cast<size_t>(row) *
                             static_cast<size_t>(max_nnz) +
                         static_cast<size_t>(slot);
      ell.col_idx[dst] = csr.col_idx[static_cast<size_t>(src)];
      ell.values[dst] = csr.values[static_cast<size_t>(src)];
    }
  }

  return ell;
}

inline NBodyState make_nbody_state(int n, uint32_t seed,
                                   float position_scale = 0.8f) {
  NBodyState state;
  state.px.resize(static_cast<size_t>(n));
  state.py.resize(static_cast<size_t>(n));
  state.pz.resize(static_cast<size_t>(n));
  state.vx.resize(static_cast<size_t>(n), 0.0f);
  state.vy.resize(static_cast<size_t>(n), 0.0f);
  state.vz.resize(static_cast<size_t>(n), 0.0f);
  state.fx.resize(static_cast<size_t>(n), 0.0f);
  state.fy.resize(static_cast<size_t>(n), 0.0f);
  state.fz.resize(static_cast<size_t>(n), 0.0f);
  state.mass.resize(static_cast<size_t>(n));

  uint32_t rng = seed;
  for (int i = 0; i < n; ++i) {
    state.px[static_cast<size_t>(i)] =
        random_float(rng, -position_scale, position_scale);
    state.py[static_cast<size_t>(i)] =
        random_float(rng, -position_scale, position_scale);
    state.pz[static_cast<size_t>(i)] =
        random_float(rng, -position_scale, position_scale);
    state.mass[static_cast<size_t>(i)] = 1.0f + 0.1f * static_cast<float>(i % 5);
  }

  return state;
}

inline NBodyState make_two_body_orbit_state() {
  NBodyState state;
  state.px = {-0.5f, 0.5f};
  state.py = {0.0f, 0.0f};
  state.pz = {0.0f, 0.0f};
  state.vx = {0.0f, 0.0f};
  state.vy = {std::sqrt(0.5f), -std::sqrt(0.5f)};
  state.vz = {0.0f, 0.0f};
  state.fx = {0.0f, 0.0f};
  state.fy = {0.0f, 0.0f};
  state.fz = {0.0f, 0.0f};
  state.mass = {1.0f, 1.0f};
  return state;
}

inline float vector_l2_norm(const std::vector<float> &values) {
  float sum = 0.0f;
  for (float value : values)
    sum += value * value;
  return std::sqrt(sum);
}

inline bool vectors_close(const std::vector<float> &lhs,
                          const std::vector<float> &rhs, float abs_eps,
                          float rel_eps) {
  if (lhs.size() != rhs.size())
    return false;
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!nearly_equal(lhs[i], rhs[i], abs_eps, rel_eps))
      return false;
  }
  return true;
}

inline bool vectors_close(const std::vector<float> &lhs,
                          const std::vector<float> &rhs,
                          float eps = 1e-4f) {
  return vectors_close(lhs, rhs, eps, eps);
}

} // namespace scicomp_test

#endif // TAPESTRY_TEST_SCICOMP_UTILS_H
