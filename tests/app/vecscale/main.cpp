#include <cstdio>

#include "vecscale.h"

int main() {
    const uint32_t N = 1024;
    const uint32_t alpha = 7;
    
    // Input array
    uint32_t A[N];
    
    // Output arrays
    uint32_t expect_B[N];
    uint32_t calculated_B[N];
    
    // Initialize input array
    for (uint32_t i = 0; i < N; i++) {
        A[i] = i % 100;
    }
    
    // Compute expected result with CPU version
    vecscale_cpu(A, alpha, expect_B, N);
    
    // Compute result with accelerator version
    vecscale_dsa(A, alpha, calculated_B, N);
    
    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (expect_B[i] != calculated_B[i]) {
            printf("vecscale: FAILED\n");
            return 1;
        }
    }
    
    printf("vecscale: PASSED\n");
    return 0;
}

