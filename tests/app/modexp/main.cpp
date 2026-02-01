#include <cstdio>

#include "modexp.h"

int main() {
    const uint32_t N = 256;
    const uint32_t modulus = 1000000007; // Large prime number
    
    // Input arrays
    uint32_t input_base[N];
    uint32_t input_exp[N];
    
    // Output arrays
    uint32_t expect_output[N];
    uint32_t calculated_output[N];
    
    // Initialize input arrays
    for (uint32_t i = 0; i < N; i++) {
        input_base[i] = (i + 1) * 123;
        input_exp[i] = (i + 1) * 7;
    }
    
    // Compute expected result with CPU version
    modexp_cpu(input_base, input_exp, expect_output, modulus, N);
    
    // Compute result with accelerator version
    modexp_dsa(input_base, input_exp, calculated_output, modulus, N);
    
    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (expect_output[i] != calculated_output[i]) {
            printf("modexp: FAILED\n");
            return 1;
        }
    }
    
    printf("modexp: PASSED\n");
    return 0;
}

