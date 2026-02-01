#include <cstdio>

#include "hash_mix.h"

int main() {
    const uint32_t N = 1024;

    // Input arrays
    uint32_t input_state[N];
    uint32_t input_data[N];

    // Output arrays
    uint32_t expect_output[N];
    uint32_t calculated_output[N];

    // Initialize input arrays
    for (uint32_t i = 0; i < N; i++) {
        input_state[i] = 0x67452301 + i; // Initial hash state
        input_data[i] = 0xEFCDAB89 + i * 13; // Data to mix
    }

    // Compute expected result with CPU version
    hash_mix_cpu(input_state, input_data, expect_output, N);

    // Compute result with accelerator version
    hash_mix_dsa(input_state, input_data, calculated_output, N);

    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (expect_output[i] != calculated_output[i]) {
            printf("hash_mix: FAILED\n");
            return 1;
        }
    }

    printf("hash_mix: PASSED\n");
    return 0;
}

