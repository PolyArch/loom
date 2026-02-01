// Loom kernel implementation: spmm
#include "spmm.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Sparse matrix-dense matrix multiplication (SpMM)
// Tests complete compilation chain with CSR format and irregular memory access
// Test: A(2x3 sparse) * B(3x2 dense) = C(2x2) → [7,10,9,12]

// CPU implementation of Sparse Matrix-Dense Matrix Multiplication (SpMM)
// Matrix A: M x N sparse matrix in CSR format
// Matrix B: N x K dense matrix (row-major)
// Matrix C: M x K dense output matrix (row-major)
// CSR format:
//   - values: non-zero values
//   - col_indices: column indices of non-zero values
//   - row_ptr: row_ptr[i] is the start of row i in values array
void spmm_cpu(const uint32_t* __restrict__ A_values,
              const uint32_t* __restrict__ A_col_indices,
              const uint32_t* __restrict__ A_row_ptr,
              const uint32_t* __restrict__ B,
              uint32_t* __restrict__ C,
              const uint32_t M,
              const uint32_t N,
              const uint32_t K) {
    // Initialize C to zero
    for (uint32_t i = 0; i < M * K; i++) {
        C[i] = 0;
    }

    // Compute C = A * B
    for (uint32_t i = 0; i < M; i++) {
        uint32_t row_start = A_row_ptr[i];
        uint32_t row_end = A_row_ptr[i + 1];

        for (uint32_t j = row_start; j < row_end; j++) {
            uint32_t A_val = A_values[j];
            uint32_t A_col = A_col_indices[j];

            // Add A[i,A_col] * B[A_col,:] to C[i,:]
            for (uint32_t k = 0; k < K; k++) {
                C[i * K + k] += A_val * B[A_col * K + k];
            }
        }
    }
}

// Accelerator implementation of SpMM
LOOM_ACCEL()
void spmm_dsa(LOOM_MEMORY_BANK(8) LOOM_STREAM const uint32_t* __restrict__ A_values,
              LOOM_STREAM const uint32_t* __restrict__ A_col_indices,
              const uint32_t* __restrict__ A_row_ptr,
              const uint32_t* __restrict__ B,
              uint32_t* __restrict__ C,
              const uint32_t M,
              const uint32_t N,
              const uint32_t K) {
    // Initialize C to zero
    for (uint32_t i = 0; i < M * K; i++) {
        C[i] = 0;
    }

    // Compute C = A * B
    for (uint32_t i = 0; i < M; i++) {
        uint32_t row_start = A_row_ptr[i];
        uint32_t row_end = A_row_ptr[i + 1];

        for (uint32_t j = row_start; j < row_end; j++) {
            uint32_t A_val = A_values[j];
            uint32_t A_col = A_col_indices[j];

            // Add A[i,A_col] * B[A_col,:] to C[i,:]
            for (uint32_t k = 0; k < K; k++) {
                C[i * K + k] += A_val * B[A_col * K + k];
            }
        }
    }
}

