#include <cstdio>

#include "byte_swap.h"

int main() {
    const uint32_t N = 256;
    
    // Input data with various byte patterns
    uint32_t input[N];
    input[0] = 0x00000000;  // All zeros
    input[1] = 0xFFFFFFFF;  // All ones
    input[2] = 0x12345678;  // -> 0x78563412
    input[3] = 0x11223344;  // -> 0x44332211
    input[4] = 0xFF000000;  // -> 0x000000FF
    input[5] = 0x000000FF;  // -> 0xFF000000
    input[6] = 0xABCDEF01;  // -> 0x01EFCDAB
    
    for (uint32_t i = 7; i < N; i++) {
        input[i] = i * 0x01020304;
    }
    
    // Output arrays
    uint32_t expect_swapped[N];
    uint32_t calculated_swapped[N];
    
    // Compute expected result with CPU version
    byte_swap_cpu(input, expect_swapped, N);
    
    // Compute result with accelerator version
    byte_swap_dsa(input, calculated_swapped, N);
    
    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (expect_swapped[i] != calculated_swapped[i]) {
            printf("byte_swap: FAILED\n");
            return 1;
        }
    }
    
    printf("byte_swap: PASSED\n");
    return 0;
}

