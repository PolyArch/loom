#include <cstdio>

#include "delta_decode.h"

int main() {
    const uint32_t N = 10;

    // Input delta encoded data (first value + differences)
    uint32_t input[N] = {100, 2, 3, 5, 5, 7, 8, 5, 7, 8};

    // Output arrays
    uint32_t expect_output[N];
    uint32_t calculated_output[N];

    // Compute expected result with CPU version
    delta_decode_cpu(input, expect_output, N);

    // Compute result with accelerator version
    delta_decode_dsa(input, calculated_output, N);

    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (expect_output[i] != calculated_output[i]) {
            printf("delta_decode: FAILED\n");
            return 1;
        }
    }

    printf("delta_decode: PASSED\n");
    return 0;
}

