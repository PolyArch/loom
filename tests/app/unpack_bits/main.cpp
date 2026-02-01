#include <cstdio>

#include "unpack_bits.h"

int main() {
    const uint32_t num_bits = 100;
    const uint32_t num_words = (num_bits + 31) / 32;

    // Input packed data
    uint32_t input_packed[num_words];
    for (uint32_t i = 0; i < num_words; i++) {
        input_packed[i] = 0xAAAAAAAA;  // Pattern: 10101010...
    }

    // Output bits
    uint32_t expect_output[num_bits];
    uint32_t calculated_output[num_bits];

    // Compute expected result with CPU version
    unpack_bits_cpu(input_packed, expect_output, num_bits);

    // Compute result with accelerator version
    unpack_bits_dsa(input_packed, calculated_output, num_bits);

    // Compare results
    for (uint32_t i = 0; i < num_bits; i++) {
        if (expect_output[i] != calculated_output[i]) {
            printf("unpack_bits: FAILED\n");
            return 1;
        }
    }

    printf("unpack_bits: PASSED\n");
    return 0;
}

