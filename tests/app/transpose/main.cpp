#include <cstdio>

#include "transpose.h"

int main() {
    const uint32_t M = 32;
    const uint32_t N = 64;

    // Input matrix (M x N)
    uint32_t A[M * N];

    // Output matrices (N x M)
    uint32_t expect_B[N * M];
    uint32_t calculated_B[N * M];

    // Initialize input matrix
    for (uint32_t i = 0; i < M * N; i++) {
        A[i] = i;
    }

    // Compute expected result with CPU version
    transpose_cpu(A, expect_B, M, N);

    // Compute result with accelerator version
    transpose_dsa(A, calculated_B, M, N);

    // Compare results
    for (uint32_t i = 0; i < N * M; i++) {
        if (expect_B[i] != calculated_B[i]) {
            printf("transpose: FAILED\n");
            return 1;
        }
    }

    printf("transpose: PASSED\n");
    return 0;
}

