#include <cstdio>

#include "gf_mul.h"

int main() {
    const uint32_t N = 256;
    
    // Input arrays
    uint32_t input_A[N];
    uint32_t input_B[N];
    
    // Output arrays
    uint32_t expect_output[N];
    uint32_t calculated_output[N];
    
    // Initialize input arrays (using all possible byte values)
    for (uint32_t i = 0; i < N; i++) {
        input_A[i] = i;
        input_B[i] = (i * 3 + 7) & 0xFF;
    }
    
    // Compute expected result with CPU version
    gf_mul_cpu(input_A, input_B, expect_output, N);
    
    // Compute result with accelerator version
    gf_mul_dsa(input_A, input_B, calculated_output, N);
    
    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (expect_output[i] != calculated_output[i]) {
            printf("gf_mul: FAILED\n");
            return 1;
        }
    }
    
    printf("gf_mul: PASSED\n");
    return 0;
}

