#include <cstdio>

#include "trsv_lower.h"

int main() {
    const uint32_t N = 16;

    // Lower triangular matrix L (with diagonal = 1 for simplicity)
    uint32_t L[N * N];
    for (uint32_t i = 0; i < N * N; i++) {
        L[i] = 0;
    }
    for (uint32_t i = 0; i < N; i++) {
        L[i * N + i] = 1;  // Diagonal
        for (uint32_t j = 0; j < i; j++) {
            L[i * N + j] = (i + j + 1) % 3;  // Lower triangular part
        }
    }

    // Right-hand side vector
    uint32_t b[N];
    for (uint32_t i = 0; i < N; i++) {
        b[i] = i + 1;
    }

    // Output vectors
    uint32_t expect_x[N];
    uint32_t calculated_x[N];

    // Test lower triangular solve
    trsv_lower_cpu(L, b, expect_x, N);
    trsv_lower_dsa(L, b, calculated_x, N);

    for (uint32_t i = 0; i < N; i++) {
        if (expect_x[i] != calculated_x[i]) {
            printf("trsv_lower: FAILED\n");
            return 1;
        }
    }

    printf("trsv_lower: PASSED\n");
    return 0;
}

