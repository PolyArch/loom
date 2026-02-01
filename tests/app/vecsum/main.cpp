#include <cstdio>

#include "vecsum.h"

int main() {
    const uint32_t N = 1024;
    const uint32_t init_value = 100;
    
    // Input array
    uint32_t A[N];
    
    // Initialize input array
    for (uint32_t i = 0; i < N; i++) {
        A[i] = i;
    }
    
    // Compute expected result with CPU version
    uint32_t expect_result = vecsum_cpu(A, init_value, N);
    
    // Compute result with accelerator version
    uint32_t calculated_result = vecsum_dsa(A, init_value, N);
    
    // Compare results
    if (expect_result != calculated_result) {
        printf("vecsum: FAILED\n");
        return 1;
    }
    
    printf("vecsum: PASSED\n");
    return 0;
}

