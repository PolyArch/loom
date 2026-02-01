#include <cstdio>

#include "rotate_bits.h"

int main() {
    const uint32_t N = 1024;
    
    // Input arrays
    uint32_t input_data[N];
    uint32_t input_shift[N];
    
    // Output arrays
    uint32_t expect_output[N];
    uint32_t calculated_output[N];
    
    // Initialize input arrays
    for (uint32_t i = 0; i < N; i++) {
        input_data[i] = 0x12345678 + i;
        input_shift[i] = i & 0x1F; // Rotation amounts from 0 to 31
    }
    
    // Compute expected result with CPU version
    rotate_bits_cpu(input_data, input_shift, expect_output, N);
    
    // Compute result with accelerator version
    rotate_bits_dsa(input_data, input_shift, calculated_output, N);
    
    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (expect_output[i] != calculated_output[i]) {
            printf("rotate_bits: FAILED\n");
            return 1;
        }
    }
    
    printf("rotate_bits: PASSED\n");
    return 0;
}

