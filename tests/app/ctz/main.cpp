#include <cstdio>

#include "ctz.h"

int main() {
    const uint32_t N = 256;

    // Input data with various bit patterns
    uint32_t input[N];
    input[0] = 0;           // Special case: all zeros
    input[1] = 0x00000001;  // LSB set: 0 trailing zeros
    input[2] = 0x00000002;  // 1 trailing zero
    input[3] = 0x80000000;  // MSB set: 31 trailing zeros
    input[4] = 0xFFFFFFFF;  // All set: 0 trailing zeros

    for (uint32_t i = 5; i < N; i++) {
        input[i] = i * 0x5678;
    }

    // Output arrays
    uint32_t expect_count[N];
    uint32_t calculated_count[N];

    // Compute expected result with CPU version
    ctz_cpu(input, expect_count, N);

    // Compute result with accelerator version
    ctz_dsa(input, calculated_count, N);

    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (expect_count[i] != calculated_count[i]) {
            printf("ctz: FAILED\n");
            return 1;
        }
    }

    printf("ctz: PASSED\n");
    return 0;
}

