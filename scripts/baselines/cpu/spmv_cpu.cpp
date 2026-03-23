// SpMV (Sparse Matrix-Vector multiply) CPU baseline using CSR format.
// Supports OpenMP parallelism across rows.

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <vector>

#include "../common/timer.h"

#ifdef _OPENMP
#include <omp.h>
#endif

// -----------------------------------------------------------------------
// CSR SpMV
// -----------------------------------------------------------------------

static void spmv_csr(const int *row_ptr, const int *col_idx,
                     const float *values, const float *x, float *y, int rows) {
#pragma omp parallel for schedule(dynamic, 64)
  for (int r = 0; r < rows; ++r) {
    float sum = 0.0f;
    for (int j = row_ptr[r]; j < row_ptr[r + 1]; ++j) {
      sum += values[j] * x[col_idx[j]];
    }
    y[r] = sum;
  }
}

// -----------------------------------------------------------------------
// Random sparse matrix generation
// -----------------------------------------------------------------------

static void generate_sparse_csr(int rows, int cols, double density,
                                std::vector<int> &row_ptr,
                                std::vector<int> &col_idx,
                                std::vector<float> &values) {
  row_ptr.resize(rows + 1);
  col_idx.clear();
  values.clear();

  row_ptr[0] = 0;
  for (int r = 0; r < rows; ++r) {
    for (int c = 0; c < cols; ++c) {
      if (static_cast<double>(rand()) / RAND_MAX < density) {
        col_idx.push_back(c);
        values.push_back(static_cast<float>(rand()) / RAND_MAX);
      }
    }
    row_ptr[r + 1] = static_cast<int>(col_idx.size());
  }
}

// -----------------------------------------------------------------------
// Main
// -----------------------------------------------------------------------

int main(int argc, char **argv) {
  int rows = 10000, cols = 10000;
  double density = 0.01;
  if (argc >= 3) {
    rows = std::atoi(argv[1]);
    cols = std::atoi(argv[2]);
  }
  if (argc >= 4) {
    density = std::atof(argv[3]);
  }

  std::vector<int> row_ptr, col_idx;
  std::vector<float> values;
  generate_sparse_csr(rows, cols, density, row_ptr, col_idx, values);

  int nnz = static_cast<int>(values.size());
  double total_ops = 2.0 * nnz;

  std::vector<float> x(cols), y(rows, 0.0f);
  for (int i = 0; i < cols; ++i)
    x[i] = static_cast<float>(rand()) / RAND_MAX;

  auto spmv_fn = [&]() {
    spmv_csr(row_ptr.data(), col_idx.data(), values.data(), x.data(), y.data(),
             rows);
  };

  auto timer = baselines::benchmark(spmv_fn);

  std::printf("RESULT kernel=spmv_csr_cpu mean_ms=%.4f stddev_ms=%.4f "
              "min_ms=%.4f total_ops=%.0f rows=%d cols=%d nnz=%d density=%.4f\n",
              timer.mean_ms(), timer.stddev_ms(), timer.min_ms(), total_ops,
              rows, cols, nnz, density);

  return 0;
}
