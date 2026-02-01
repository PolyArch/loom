#include <cstdio>

#include "xor_block.h"

int main() {
    const uint32_t N = 1024;

    // Input arrays
    uint32_t input_A[N];
    uint32_t input_B[N];

    // Output arrays
    uint32_t expect_output[N];
    uint32_t calculated_output[N];

    // Initialize input arrays
    for (uint32_t i = 0; i < N; i++) {
        input_A[i] = i * 0x12345678;
        input_B[i] = i * 0xABCDEF01;
    }

    // Compute expected result with CPU version
    xor_block_cpu(input_A, input_B, expect_output, N);

    // Compute result with accelerator version
    xor_block_dsa(input_A, input_B, calculated_output, N);

    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (expect_output[i] != calculated_output[i]) {
            printf("xor_block: FAILED\n");
            return 1;
        }
    }

    printf("xor_block: PASSED\n");
    return 0;
}

