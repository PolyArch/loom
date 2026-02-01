#include <cstdio>

#include "outer.h"

int main() {
    const uint32_t M = 32;
    const uint32_t N = 48;
    
    // Input vectors
    uint32_t a[M];
    uint32_t b[N];
    
    // Output matrices (M x N)
    uint32_t expect_C[M * N];
    uint32_t calculated_C[M * N];
    
    // Initialize input vectors
    for (uint32_t i = 0; i < M; i++) {
        a[i] = i + 1;
    }
    for (uint32_t i = 0; i < N; i++) {
        b[i] = (i * 2) + 1;
    }
    
    // Compute expected result with CPU version
    outer_cpu(a, b, expect_C, M, N);
    
    // Compute result with accelerator version
    outer_dsa(a, b, calculated_C, M, N);
    
    // Compare results
    for (uint32_t i = 0; i < M * N; i++) {
        if (expect_C[i] != calculated_C[i]) {
            printf("outer: FAILED\n");
            return 1;
        }
    }
    
    printf("outer: PASSED\n");
    return 0;
}

