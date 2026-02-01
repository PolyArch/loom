#include <cstdio>

#include "prefix_sum_inclusive.h"

int main() {
    const uint32_t N = 1024;

    // Input array
    uint32_t input[N];

    // Output arrays
    uint32_t expect_output[N];
    uint32_t calculated_output[N];

    // Initialize input
    for (uint32_t i = 0; i < N; i++) {
        input[i] = (i % 10) + 1;
    }

    // Test inclusive prefix sum
    prefix_sum_inclusive_cpu(input, expect_output, N);
    prefix_sum_inclusive_dsa(input, calculated_output, N);

    for (uint32_t i = 0; i < N; i++) {
        if (expect_output[i] != calculated_output[i]) {
            printf("prefix_sum_inclusive: FAILED\n");
            return 1;
        }
    }

    printf("prefix_sum_inclusive: PASSED\n");
    return 0;
}

