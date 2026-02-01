// Loom app test driver: spmspm
#include "spmspm.h"
#include <cstdio>

int main() {
    // Test case: A (3x4) * B (4x3) = C (3x3)
    const uint32_t M = 3;
    const uint32_t N = 4;
    const uint32_t K = 3;

    // Matrix A (3x4): [ 2 0 1 0 ], [ 0 3 0 2 ], [ 1 0 0 4 ]
    const uint32_t A_nnz = 6;
    uint32_t A_values[A_nnz] = {2, 1, 3, 2, 1, 4};
    uint32_t A_col_indices[A_nnz] = {0, 2, 1, 3, 0, 3};
    uint32_t A_row_ptr[M + 1] = {0, 2, 4, 6};

    // Matrix B (4x3): [ 1 0 2 ], [ 0 5 0 ], [ 3 0 1 ], [ 0 2 0 ]
    const uint32_t B_nnz = 6;
    uint32_t B_values[B_nnz] = {1, 2, 5, 3, 1, 2};
    uint32_t B_col_indices[B_nnz] = {0, 2, 1, 0, 2, 1};
    uint32_t B_row_ptr[N + 1] = {0, 2, 3, 5, 6};

    // Output buffers
    const uint32_t C_max_nnz = M * K;
    uint32_t expect_C_values[C_max_nnz] = {0};
    uint32_t expect_C_col_indices[C_max_nnz] = {0};
    uint32_t expect_C_row_ptr[M + 1] = {0};

    uint32_t calculated_C_values[C_max_nnz] = {0};
    uint32_t calculated_C_col_indices[C_max_nnz] = {0};
    uint32_t calculated_C_row_ptr[M + 1] = {0};

    // Temporary buffers
    uint32_t cpu_temp_row[K] = {0};
    uint32_t dsa_temp_row[K] = {0};

    // Compute expected result with CPU version
    spmspm_cpu(A_values, A_col_indices, A_row_ptr,
               B_values, B_col_indices, B_row_ptr,
               expect_C_values, expect_C_col_indices, expect_C_row_ptr,
               cpu_temp_row, M, N, K);

    // Compute result with DSA version
    spmspm_dsa(A_values, A_col_indices, A_row_ptr,
               B_values, B_col_indices, B_row_ptr,
               calculated_C_values, calculated_C_col_indices, calculated_C_row_ptr,
               dsa_temp_row, M, N, K);

    // Compare results
    bool passed = true;
    for (uint32_t i = 0; i <= M; i++) {
        if (expect_C_row_ptr[i] != calculated_C_row_ptr[i]) {
            passed = false;
            break;
        }
    }

    if (passed) {
        uint32_t C_nnz = expect_C_row_ptr[M];
        for (uint32_t i = 0; i < C_nnz; i++) {
            if (expect_C_values[i] != calculated_C_values[i] ||
                expect_C_col_indices[i] != calculated_C_col_indices[i]) {
                passed = false;
                break;
            }
        }
    }

    if (passed) {
        printf("spmspm: PASSED\n");
        return 0;
    } else {
        printf("spmspm: FAILED\n");
        return 1;
    }
}
