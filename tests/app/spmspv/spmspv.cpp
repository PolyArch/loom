// Loom kernel implementation: spmspv
#include "spmspv.h"
#include "loom/loom.h"

void spmspv_cpu(const uint32_t* __restrict__ A_values,
                const uint32_t* __restrict__ A_col_indices,
                const uint32_t* __restrict__ A_row_ptr,
                const uint32_t* __restrict__ x_values,
                const uint32_t* __restrict__ x_indices,
                const uint32_t x_nnz,
                uint32_t* __restrict__ y,
                uint32_t* __restrict__ x_dense,
                const uint32_t M,
                const uint32_t N) {
    // Initialize x_dense to zero
    for (uint32_t i = 0; i < N; i++) {
        x_dense[i] = 0;
    }

    // Populate x_dense from sparse representation
    for (uint32_t i = 0; i < x_nnz; i++) {
        x_dense[x_indices[i]] = x_values[i];
    }

    // Compute y = A * x
    for (uint32_t i = 0; i < M; i++) {
        uint32_t sum = 0;
        uint32_t row_start = A_row_ptr[i];
        uint32_t row_end = A_row_ptr[i + 1];

        for (uint32_t j = row_start; j < row_end; j++) {
            uint32_t col = A_col_indices[j];
            sum += A_values[j] * x_dense[col];
        }

        y[i] = sum;
    }
}

LOOM_ACCEL()
void spmspv_dsa(LOOM_MEMORY_BANK(8) LOOM_STREAM const uint32_t* __restrict__ A_values,
                LOOM_STREAM const uint32_t* __restrict__ A_col_indices,
                const uint32_t* __restrict__ A_row_ptr,
                const uint32_t* __restrict__ x_values,
                const uint32_t* __restrict__ x_indices,
                const uint32_t x_nnz,
                uint32_t* __restrict__ y,
                uint32_t* __restrict__ x_dense,
                const uint32_t M,
                const uint32_t N) {
    // Initialize x_dense to zero
    for (uint32_t i = 0; i < N; i++) {
        x_dense[i] = 0;
    }

    // Populate x_dense from sparse representation
    for (uint32_t i = 0; i < x_nnz; i++) {
        x_dense[x_indices[i]] = x_values[i];
    }

    // Compute y = A * x
    for (uint32_t i = 0; i < M; i++) {
        uint32_t sum = 0;
        uint32_t row_start = A_row_ptr[i];
        uint32_t row_end = A_row_ptr[i + 1];

        for (uint32_t j = row_start; j < row_end; j++) {
            uint32_t col = A_col_indices[j];
            sum += A_values[j] * x_dense[col];
        }

        y[i] = sum;
    }
}
