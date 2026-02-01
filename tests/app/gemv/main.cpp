#include <cstdio>

#include "gemv.h"

int main() {
    const uint32_t M = 32;
    const uint32_t N = 48;
    const uint32_t alpha = 2;
    const uint32_t beta = 3;
    
    // Input matrix and vectors
    uint32_t A[M * N];
    uint32_t x[N];
    uint32_t input_y[M];
    
    // Output vectors (separate for CPU and DSA)
    uint32_t expect_y[M];
    uint32_t calculated_y[M];
    
    // Initialize inputs
    for (uint32_t i = 0; i < M * N; i++) {
        A[i] = (i % 10) + 1;
    }
    for (uint32_t i = 0; i < N; i++) {
        x[i] = i % 7;
    }
    for (uint32_t i = 0; i < M; i++) {
        input_y[i] = i % 5;
    }
    
    // Compute expected result with CPU version
    gemv_cpu(alpha, A, x, beta, input_y, expect_y, M, N);
    
    // Compute result with accelerator version
    gemv_dsa(alpha, A, x, beta, input_y, calculated_y, M, N);
    
    // Compare results
    for (uint32_t i = 0; i < M; i++) {
        if (expect_y[i] != calculated_y[i]) {
            printf("gemv: FAILED\n");
            return 1;
        }
    }
    
    printf("gemv: PASSED\n");
    return 0;
}

