#include <cstdio>

#include "rle_encode.h"

int main() {
    const uint32_t N = 20;

    // Input array with runs
    uint32_t input[N] = {
        1, 1, 1, 2, 2, 3, 3, 3, 3, 4,
        4, 4, 4, 4, 5, 6, 6, 6, 7, 7
    };

    // Output arrays (worst case: each element is unique, so N elements)
    uint32_t expect_values[N];
    uint32_t expect_counts[N];
    uint32_t expect_length;

    uint32_t calculated_values[N];
    uint32_t calculated_counts[N];
    uint32_t calculated_length;

    // Compute expected result with CPU version
    rle_encode_cpu(input, expect_values, expect_counts, &expect_length, N);

    // Compute result with accelerator version
    rle_encode_dsa(input, calculated_values, calculated_counts, &calculated_length, N);

    // Compare lengths
    if (expect_length != calculated_length) {
        printf("rle_encode: FAILED\n");
        return 1;
    }

    // Compare results
    for (uint32_t i = 0; i < expect_length; i++) {
        if (expect_values[i] != calculated_values[i] ||
            expect_counts[i] != calculated_counts[i]) {
            printf("rle_encode: FAILED\n");
            return 1;
        }
    }

    printf("rle_encode: PASSED\n");
    return 0;
}

