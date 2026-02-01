#include <cstdio>

#include "trsv_upper.h"

int main() {
    const uint32_t N = 16;
    
    // Upper triangular matrix U (with diagonal = 1 for simplicity)
    uint32_t U[N * N];
    for (uint32_t i = 0; i < N * N; i++) {
        U[i] = 0;
    }
    for (uint32_t i = 0; i < N; i++) {
        U[i * N + i] = 1;  // Diagonal
        for (uint32_t j = i + 1; j < N; j++) {
            U[i * N + j] = (i + j + 1) % 3;  // Upper triangular part
        }
    }
    
    // Right-hand side vector
    uint32_t b[N];
    for (uint32_t i = 0; i < N; i++) {
        b[i] = N - i;
    }
    
    // Output vectors
    uint32_t expect_x[N];
    uint32_t calculated_x[N];
    
    // Test upper triangular solve
    trsv_upper_cpu(U, b, expect_x, N);
    trsv_upper_dsa(U, b, calculated_x, N);
    
    for (uint32_t i = 0; i < N; i++) {
        if (expect_x[i] != calculated_x[i]) {
            printf("trsv_upper: FAILED\n");
            return 1;
        }
    }
    
    printf("trsv_upper: PASSED\n");
    return 0;
}

