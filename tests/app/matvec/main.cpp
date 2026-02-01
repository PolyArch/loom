#include <cstdio>

#include "matvec.h"

int main() {
    const uint32_t M = 32;
    const uint32_t N = 48;
    
    // Input matrix and vector
    uint32_t A[M * N];
    uint32_t x[N];
    
    // Output vectors
    uint32_t expect_y[M];
    uint32_t calculated_y[M];
    
    // Initialize inputs
    for (uint32_t i = 0; i < M * N; i++) {
        A[i] = (i % 10) + 1;
    }
    for (uint32_t i = 0; i < N; i++) {
        x[i] = i % 7;
    }
    
    // Compute expected result with CPU version
    matvec_cpu(A, x, expect_y, M, N);
    
    // Compute result with accelerator version
    matvec_dsa(A, x, calculated_y, M, N);
    
    // Compare results
    for (uint32_t i = 0; i < M; i++) {
        if (expect_y[i] != calculated_y[i]) {
            printf("matvec: FAILED\n");
            return 1;
        }
    }
    
    printf("matvec: PASSED\n");
    return 0;
}

