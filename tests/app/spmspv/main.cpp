// Loom app test driver: spmspv
#include "spmspv.h"
#include <cstdio>

int main() {
    // Test case: A (4x5) * x (sparse vec) = y (dense vec)
    const uint32_t M = 4;
    const uint32_t N = 5;

    // Matrix A (4x5): 9 non-zeros
    const uint32_t A_nnz = 9;
    uint32_t A_values[A_nnz] = {2, 3, 4, 1, 5, 6, 7, 2, 3};
    uint32_t A_col_indices[A_nnz] = {0, 2, 1, 3, 0, 4, 1, 2, 4};
    uint32_t A_row_ptr[M + 1] = {0, 2, 4, 6, 9};

    // Sparse vector x: [3,0,2,5,0] with 3 non-zeros
    const uint32_t x_nnz = 3;
    uint32_t x_values[x_nnz] = {3, 2, 5};
    uint32_t x_indices[x_nnz] = {0, 2, 3};

    // Output vectors
    uint32_t expect_y[M] = {0};
    uint32_t calculated_y[M] = {0};

    // Temporary buffers for x_dense
    uint32_t cpu_x_dense[N] = {0};
    uint32_t dsa_x_dense[N] = {0};

    // Compute expected result with CPU version
    spmspv_cpu(A_values, A_col_indices, A_row_ptr,
               x_values, x_indices, x_nnz,
               expect_y, cpu_x_dense, M, N);

    // Compute result with DSA version
    spmspv_dsa(A_values, A_col_indices, A_row_ptr,
               x_values, x_indices, x_nnz,
               calculated_y, dsa_x_dense, M, N);

    // Compare results
    bool passed = true;
    for (uint32_t i = 0; i < M; i++) {
        if (expect_y[i] != calculated_y[i]) {
            passed = false;
            break;
        }
    }

    if (passed) {
        printf("spmspv: PASSED\n");
        return 0;
    } else {
        printf("spmspv: FAILED\n");
        return 1;
    }
}
