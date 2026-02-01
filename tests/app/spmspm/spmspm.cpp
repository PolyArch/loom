// Loom kernel implementation: spmspm
#include "spmspm.h"
#include "loom/loom.h"

void spmspm_cpu(const uint32_t* __restrict__ A_values,
                const uint32_t* __restrict__ A_col_indices,
                const uint32_t* __restrict__ A_row_ptr,
                const uint32_t* __restrict__ B_values,
                const uint32_t* __restrict__ B_col_indices,
                const uint32_t* __restrict__ B_row_ptr,
                uint32_t* __restrict__ C_values,
                uint32_t* __restrict__ C_col_indices,
                uint32_t* __restrict__ C_row_ptr,
                uint32_t* __restrict__ temp_row,
                const uint32_t M,
                const uint32_t N,
                const uint32_t K) {
    uint32_t nnz = 0;
    C_row_ptr[0] = 0;

    for (uint32_t i = 0; i < M; i++) {
        // Initialize temporary row to zero
        for (uint32_t j = 0; j < K; j++) {
            temp_row[j] = 0;
        }

        // Compute row i of C
        uint32_t A_row_start = A_row_ptr[i];
        uint32_t A_row_end = A_row_ptr[i + 1];

        for (uint32_t j = A_row_start; j < A_row_end; j++) {
            uint32_t A_val = A_values[j];
            uint32_t A_col = A_col_indices[j];

            // Add A[i,A_col] * B[A_col,:] to temp_row
            uint32_t B_row_start = B_row_ptr[A_col];
            uint32_t B_row_end = B_row_ptr[A_col + 1];

            for (uint32_t k = B_row_start; k < B_row_end; k++) {
                uint32_t B_col = B_col_indices[k];
                temp_row[B_col] += A_val * B_values[k];
            }
        }

        // Compress temp_row into CSR format
        for (uint32_t j = 0; j < K; j++) {
            if (temp_row[j] != 0) {
                C_values[nnz] = temp_row[j];
                C_col_indices[nnz] = j;
                nnz++;
            }
        }

        C_row_ptr[i + 1] = nnz;
    }
}

LOOM_ACCEL()
void spmspm_dsa(LOOM_MEMORY_BANK(8) LOOM_STREAM const uint32_t* __restrict__ A_values,
                LOOM_STREAM const uint32_t* __restrict__ A_col_indices,
                const uint32_t* __restrict__ A_row_ptr,
                const uint32_t* __restrict__ B_values,
                const uint32_t* __restrict__ B_col_indices,
                const uint32_t* __restrict__ B_row_ptr,
                uint32_t* __restrict__ C_values,
                uint32_t* __restrict__ C_col_indices,
                uint32_t* __restrict__ C_row_ptr,
                uint32_t* __restrict__ temp_row,
                const uint32_t M,
                const uint32_t N,
                const uint32_t K) {
    uint32_t nnz = 0;
    C_row_ptr[0] = 0;

    for (uint32_t i = 0; i < M; i++) {
        // Initialize temporary row to zero
        for (uint32_t j = 0; j < K; j++) {
            temp_row[j] = 0;
        }

        // Compute row i of C
        uint32_t A_row_start = A_row_ptr[i];
        uint32_t A_row_end = A_row_ptr[i + 1];

        for (uint32_t j = A_row_start; j < A_row_end; j++) {
            uint32_t A_val = A_values[j];
            uint32_t A_col = A_col_indices[j];

            // Add A[i,A_col] * B[A_col,:] to temp_row
            uint32_t B_row_start = B_row_ptr[A_col];
            uint32_t B_row_end = B_row_ptr[A_col + 1];

            for (uint32_t k = B_row_start; k < B_row_end; k++) {
                uint32_t B_col = B_col_indices[k];
                temp_row[B_col] += A_val * B_values[k];
            }
        }

        // Compress temp_row into CSR format
        for (uint32_t j = 0; j < K; j++) {
            if (temp_row[j] != 0) {
                C_values[nnz] = temp_row[j];
                C_col_indices[nnz] = j;
                nnz++;
            }
        }

        C_row_ptr[i + 1] = nnz;
    }
}
