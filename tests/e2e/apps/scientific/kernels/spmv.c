#include "scicomp_types.h"

void spmv_csr(const int *row_ptr, const int *col_idx, const float *values,
              const float *x, float *y, int nrows) {
  if (!row_ptr || !col_idx || !values || !x || !y || nrows <= 0)
    return;

  for (int row = 0; row < nrows; ++row) {
    float sum = 0.0f;
    for (int idx = row_ptr[row]; idx < row_ptr[row + 1]; ++idx)
      sum += values[idx] * x[col_idx[idx]];
    y[row] = sum;
  }
}

void spmv_ell(const int *col_idx, const float *values, const float *x, float *y,
              int nrows, int max_nnz) {
  if (!col_idx || !values || !x || !y || nrows <= 0 || max_nnz <= 0)
    return;

  for (int row = 0; row < nrows; ++row) {
    float sum = 0.0f;
    for (int slot = 0; slot < max_nnz; ++slot) {
      int idx = row * max_nnz + slot;
      int col = col_idx[idx];
      if (col >= 0)
        sum += values[idx] * x[col];
    }
    y[row] = sum;
  }
}

void spmv_csr_unroll2(const int *row_ptr, const int *col_idx,
                      const float *values, const float *x, float *y,
                      int nrows) {
  if (!row_ptr || !col_idx || !values || !x || !y || nrows <= 0)
    return;

  for (int row = 0; row < nrows; ++row) {
    float sum0 = 0.0f;
    int idx = row_ptr[row];
    int end = row_ptr[row + 1];
    for (; idx + 1 < end; idx += 2) {
      sum0 += values[idx] * x[col_idx[idx]];
      sum0 += values[idx + 1] * x[col_idx[idx + 1]];
    }
    for (; idx < end; ++idx)
      sum0 += values[idx] * x[col_idx[idx]];
    y[row] = sum0;
  }
}
