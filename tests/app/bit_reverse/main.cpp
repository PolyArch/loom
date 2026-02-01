#include <cstdio>

#include "bit_reverse.h"

int main() {
    const uint32_t N = 256;

    // Input data with various bit patterns
    uint32_t input[N];
    input[0] = 0x00000000;  // All zeros
    input[1] = 0xFFFFFFFF;  // All ones
    input[2] = 0x80000000;  // MSB set -> LSB set
    input[3] = 0x00000001;  // LSB set -> MSB set
    input[4] = 0xF0F0F0F0;  // Pattern
    input[5] = 0x12345678;  // Random pattern

    for (uint32_t i = 6; i < N; i++) {
        input[i] = i * 0xABCD1234;
    }

    // Output arrays
    uint32_t expect_reversed[N];
    uint32_t calculated_reversed[N];

    // Compute expected result with CPU version
    bit_reverse_cpu(input, expect_reversed, N);

    // Compute result with accelerator version
    bit_reverse_dsa(input, calculated_reversed, N);

    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (expect_reversed[i] != calculated_reversed[i]) {
            printf("bit_reverse: FAILED\n");
            return 1;
        }
    }

    printf("bit_reverse: PASSED\n");
    return 0;
}

