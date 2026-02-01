#include <cstdio>

#include "modmul.h"

int main() {
    const uint32_t N = 1024;
    const uint32_t modulus = 1000000007; // Large prime number

    // Input arrays
    uint32_t input_A[N];
    uint32_t input_B[N];

    // Output arrays
    uint32_t expect_output[N];
    uint32_t calculated_output[N];

    // Initialize input arrays
    for (uint32_t i = 0; i < N; i++) {
        input_A[i] = (i + 1) * 12345;
        input_B[i] = (i + 1) * 67890;
    }

    // Compute expected result with CPU version
    modmul_cpu(input_A, input_B, expect_output, modulus, N);

    // Compute result with accelerator version
    modmul_dsa(input_A, input_B, calculated_output, modulus, N);

    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (expect_output[i] != calculated_output[i]) {
            printf("modmul: FAILED\n");
            return 1;
        }
    }

    printf("modmul: PASSED\n");
    return 0;
}

