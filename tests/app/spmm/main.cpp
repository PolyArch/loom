#include <algorithm>
#include <cstdio>

#include "spmm.h"

int main() {
    const uint32_t M = 64;  // A: M x N
    const uint32_t N = 64;  // A: M x N, B: N x K
    const uint32_t K = 32;  // B: N x K, C: M x K

    // Sparse matrix A in CSR format (approximately 4 non-zeros per row)
    const uint32_t A_nnz = 256;
    uint32_t A_values[A_nnz];
    uint32_t A_col_indices[A_nnz];
    uint32_t A_row_ptr[M + 1];

    // Dense matrix B (N x K)
    uint32_t B[N * K];

    // Dense output matrices C (M x K)
    uint32_t expect_C[M * K];
    uint32_t calculated_C[M * K];

    // Build sparse matrix A (diagonal + some off-diagonal elements)
    uint32_t A_count = 0;
    A_row_ptr[0] = 0;
    for (uint32_t i = 0; i < M; i++) {
        uint32_t nnz_per_row = std::min(4u, A_nnz - A_count);

        // Diagonal element
        if (A_count < A_nnz && i < N) {
            A_values[A_count] = i % 10 + 1;
            A_col_indices[A_count] = i;
            A_count++;
        }

        // Off-diagonal elements
        for (uint32_t k = 1; k < nnz_per_row && A_count < A_nnz; k++) {
            uint32_t col = (i + k * 16) % N;
            A_values[A_count] = (i + k) % 10 + 1;
            A_col_indices[A_count] = col;
            A_count++;
        }

        A_row_ptr[i + 1] = A_count;
    }

    // Initialize dense matrix B
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < K; j++) {
            B[i * K + j] = (i + j) % 10 + 1;
        }
    }

    // Compute expected result with CPU version
    spmm_cpu(A_values, A_col_indices, A_row_ptr, B, expect_C, M, N, K);

    // Compute result with accelerator version
    spmm_dsa(A_values, A_col_indices, A_row_ptr, B, calculated_C, M, N, K);

    // Compare results
    for (uint32_t i = 0; i < M * K; i++) {
        if (expect_C[i] != calculated_C[i]) {
            printf("spmm: FAILED\n");
            return 1;
        }
    }

    printf("spmm: PASSED\n");
    return 0;
}
