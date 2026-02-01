#include <cstdio>

#include "find_first_set.h"

int main() {
    const uint32_t N = 256;
    
    // Input data with various bit patterns
    uint32_t input[N];
    input[0] = 0x00000000;  // No bits set -> 0
    input[1] = 0x00000001;  // LSB set -> position 1
    input[2] = 0x00000002;  // Bit 1 set -> position 2
    input[3] = 0x00000004;  // Bit 2 set -> position 3
    input[4] = 0x80000000;  // MSB set -> position 32
    input[5] = 0xFFFFFFFF;  // All set -> position 1 (LSB)
    input[6] = 0xFFFFFFF0;  // Lower 4 bits clear -> position 5
    
    for (uint32_t i = 7; i < N; i++) {
        input[i] = i * 0x8765;
    }
    
    // Output arrays
    uint32_t expect_position[N];
    uint32_t calculated_position[N];
    
    // Compute expected result with CPU version
    find_first_set_cpu(input, expect_position, N);
    
    // Compute result with accelerator version
    find_first_set_dsa(input, calculated_position, N);
    
    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (expect_position[i] != calculated_position[i]) {
            printf("find_first_set: FAILED\n");
            return 1;
        }
    }
    
    printf("find_first_set: PASSED\n");
    return 0;
}

