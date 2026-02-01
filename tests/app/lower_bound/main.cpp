#include <cstdio>

#include "lower_bound.h"

int main() {
    const uint32_t N = 10;
    const uint32_t M = 6;

    // Sorted input array
    float input_sorted[N] = {1.0f, 3.0f, 3.0f, 5.0f, 7.0f, 9.0f, 11.0f, 13.0f, 15.0f, 17.0f};

    // Target values
    float input_targets[M] = {3.0f, 0.0f, 8.0f, 20.0f, 5.0f, 11.0f};

    // Output arrays
    uint32_t expect_indices[M];
    uint32_t calculated_indices[M];

    // Compute expected result with CPU version
    lower_bound_cpu(input_sorted, input_targets, expect_indices, N, M);

    // Compute result with DSA version
    lower_bound_dsa(input_sorted, input_targets, calculated_indices, N, M);

    // Compare results
    for (uint32_t i = 0; i < M; i++) {
        if (expect_indices[i] != calculated_indices[i]) {
            printf("lower_bound: FAILED\n");
            return 1;
        }
    }

    printf("lower_bound: PASSED\n");
    return 0;
}

