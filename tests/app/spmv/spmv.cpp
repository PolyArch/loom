// Loom kernel implementation: spmv
#include "spmv.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Sparse Matrix-Vector Multiplication (SpMV) in CSR format
// Tests complete compilation chain with variable-length inner loops, indirect indexing, and per-row reduction
// Test: A(5x6 sparse CSR, 10 nnz) * x(6x1) = y(5x1) → [14, 20, 28, 20, 38]

// CPU implementation of Sparse Matrix-Vector Multiplication (SpMV)
// Matrix A: M x N sparse matrix in CSR format
// Vector x: dense vector of length N
// Vector y: dense output vector of length M
// CSR format:
//   - values: non-zero values
//   - col_indices: column indices of non-zero values
//   - row_ptr: row_ptr[i] is the start of row i in values array
//              row_ptr[M] is the total number of non-zeros
void spmv_cpu(const uint32_t* __restrict__ values,
              const uint32_t* __restrict__ col_indices,
              const uint32_t* __restrict__ row_ptr,
              const uint32_t* __restrict__ x,
              uint32_t* __restrict__ y,
              const uint32_t M,
              const uint32_t N) {
    for (uint32_t i = 0; i < M; i++) {
        uint32_t sum = 0;
        uint32_t row_start = row_ptr[i];
        uint32_t row_end = row_ptr[i + 1];

        for (uint32_t j = row_start; j < row_end; j++) {
            sum += values[j] * x[col_indices[j]];
        }

        y[i] = sum;
    }
}

// SpMV (CSR): y[i] = sum(values[j] * x[col_indices[j]]) for j in [row_ptr[i], row_ptr[i+1])
// Accelerator implementation of SpMV
LOOM_ACCEL()
void spmv_dsa(LOOM_MEMORY_BANK(8) LOOM_STREAM const uint32_t* __restrict__ values,
              LOOM_STREAM const uint32_t* __restrict__ col_indices,
              const uint32_t* __restrict__ row_ptr,
              const uint32_t* __restrict__ x,
              uint32_t* __restrict__ y,
              const uint32_t M,
              const uint32_t N) {
    for (uint32_t i = 0; i < M; i++) {
        uint32_t sum = 0;
        uint32_t row_start = row_ptr[i];
        uint32_t row_end = row_ptr[i + 1];

        for (uint32_t j = row_start; j < row_end; j++) {
            sum += values[j] * x[col_indices[j]];
        }

        y[i] = sum;
    }
}

