#include <cstdio>

#include "partition.h"
#include <cmath>

int main() {
    const uint32_t N = 10;
    const float pivot = 5.5f;

    // Input array
    float input[N] = {3.0f, 7.0f, 1.0f, 9.0f, 5.0f, 2.0f, 8.0f, 4.0f, 6.0f, 10.0f};

    // Output arrays
    float expect_output[N];
    float calculated_output[N];
    uint32_t expect_pivot_idx;
    uint32_t calculated_pivot_idx;

    // Compute expected result with CPU version
    partition_cpu(input, expect_output, &expect_pivot_idx, N, pivot);

    // Compute result with DSA version
    partition_dsa(input, calculated_output, &calculated_pivot_idx, N, pivot);

    // Compare pivot indices
    if (expect_pivot_idx != calculated_pivot_idx) {
        printf("partition: FAILED\n");
        return 1;
    }

    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > 1e-5f) {
            printf("partition: FAILED\n");
            return 1;
        }
    }

    printf("partition: PASSED\n");
    return 0;
}

