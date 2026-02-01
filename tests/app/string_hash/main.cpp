#include <cstdio>

#include "string_hash.h"

int main() {
    const uint32_t N = 128;
    const uint32_t window_size = 8;
    const uint32_t num_windows = N - window_size + 1;

    // Input string
    uint32_t input[N];
    for (uint32_t i = 0; i < N; i++) {
        input[i] = 'a' + (i % 26);
    }

    // Output hash arrays
    uint32_t expect_hashes[num_windows];
    uint32_t calculated_hashes[num_windows];

    // Compute expected result with CPU version
    string_hash_cpu(input, expect_hashes, N, window_size);

    // Compute result with accelerator version
    string_hash_dsa(input, calculated_hashes, N, window_size);

    // Compare results
    for (uint32_t i = 0; i < num_windows; i++) {
        if (expect_hashes[i] != calculated_hashes[i]) {
            printf("string_hash: FAILED\n");
            return 1;
        }
    }

    printf("string_hash: PASSED\n");
    return 0;
}

