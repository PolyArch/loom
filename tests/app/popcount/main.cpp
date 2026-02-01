#include <cstdio>

#include "popcount.h"

int main() {
    const uint32_t N = 256;
    
    // Input data with various bit patterns
    uint32_t input[N];
    for (uint32_t i = 0; i < N; i++) {
        input[i] = i * 0x12345678 + (i << 16);
    }
    
    // Output arrays
    uint32_t expect_count[N];
    uint32_t calculated_count[N];
    
    // Compute expected result with CPU version
    popcount_cpu(input, expect_count, N);
    
    // Compute result with accelerator version
    popcount_dsa(input, calculated_count, N);
    
    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (expect_count[i] != calculated_count[i]) {
            printf("popcount: FAILED\n");
            return 1;
        }
    }
    
    printf("popcount: PASSED\n");
    return 0;
}

