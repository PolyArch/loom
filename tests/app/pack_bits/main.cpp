#include <cstdio>

#include "pack_bits.h"

int main() {
    const uint32_t num_bits = 100;
    const uint32_t num_words = (num_bits + 31) / 32;

    // Input bits (0 or 1 per element)
    uint32_t input_bits[num_bits];
    for (uint32_t i = 0; i < num_bits; i++) {
        input_bits[i] = (i % 3 == 0) ? 1 : 0;
    }

    // Output packed words
    uint32_t expect_output[num_words];
    uint32_t calculated_output[num_words];

    // Compute expected result with CPU version
    pack_bits_cpu(input_bits, expect_output, num_bits);

    // Compute result with accelerator version
    pack_bits_dsa(input_bits, calculated_output, num_bits);

    // Compare results
    for (uint32_t i = 0; i < num_words; i++) {
        if (expect_output[i] != calculated_output[i]) {
            printf("pack_bits: FAILED\n");
            return 1;
        }
    }

    printf("pack_bits: PASSED\n");
    return 0;
}

