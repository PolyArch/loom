#include <cstdio>

#include "lz77_compress.h"

int main() {
    const uint32_t N = 20;
    const uint32_t window_size = 10;

    // Input array with repeated patterns for compression
    uint32_t input[N] = {
        1, 2, 3, 1, 2, 3, 4, 5, 4, 5,
        4, 5, 6, 7, 8, 6, 7, 8, 9, 9
    };

    // Output arrays (worst case: each element is a literal, so N triples)
    uint32_t expect_offsets[N];
    uint32_t expect_lengths[N];
    uint32_t expect_literals[N];
    uint32_t expect_count;

    uint32_t calculated_offsets[N];
    uint32_t calculated_lengths[N];
    uint32_t calculated_literals[N];
    uint32_t calculated_count;

    // Compute expected result with CPU version
    lz77_compress_cpu(input, expect_offsets, expect_lengths, expect_literals,
                      &expect_count, N, window_size);

    // Compute result with accelerator version
    lz77_compress_dsa(input, calculated_offsets, calculated_lengths,
                      calculated_literals, &calculated_count, N, window_size);

    // Compare output counts
    if (expect_count != calculated_count) {
        printf("lz77_compress: FAILED (count mismatch: expected %u, got %u)\n",
               expect_count, calculated_count);
        return 1;
    }

    // Compare results element-by-element
    for (uint32_t i = 0; i < expect_count; i++) {
        if (expect_offsets[i] != calculated_offsets[i] ||
            expect_lengths[i] != calculated_lengths[i] ||
            expect_literals[i] != calculated_literals[i]) {
            printf("lz77_compress: FAILED (mismatch at triple %u)\n", i);
            return 1;
        }
    }

    printf("lz77_compress: PASSED\n");
    return 0;
}
