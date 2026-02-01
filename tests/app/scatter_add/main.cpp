#include <cstdio>

#include "scatter_add.h"

int main() {
    const uint32_t N = 1024;
    const uint32_t dst_size = 256;

    // Input arrays
    uint32_t src[N];
    uint32_t indices[N];

    // Output arrays (initialized to zero)
    uint32_t expect_dst[dst_size];
    uint32_t calculated_dst[dst_size];

    // Initialize inputs
    for (uint32_t i = 0; i < N; i++) {
        src[i] = i % 10;
        indices[i] = (i * 7) % dst_size;  // Scattered indices
    }

    for (uint32_t i = 0; i < dst_size; i++) {
        expect_dst[i] = 0;
        calculated_dst[i] = 0;
    }

    // Compute expected result with CPU version
    scatter_add_cpu(src, indices, expect_dst, N, dst_size);

    // Compute result with accelerator version
    scatter_add_dsa(src, indices, calculated_dst, N, dst_size);

    // Compare results
    for (uint32_t i = 0; i < dst_size; i++) {
        if (expect_dst[i] != calculated_dst[i]) {
            printf("scatter_add: FAILED\n");
            return 1;
        }
    }

    printf("scatter_add: PASSED\n");
    return 0;
}

