#include <cstdio>

#include "sort_insertion.h"
#include <cmath>

int main() {
    const uint32_t N = 512;

    // Allocate and initialize input
    float input[N];
    for (uint32_t i = 0; i < N; i++) {
        input[i] = static_cast<float>(N - i);  // Reverse order
    }

    // Allocate output arrays
    float expect_output[N];
    float calculated_output[N];

    // Compute expected result with CPU version
    sort_insertion_cpu(input, expect_output, N);

    // Compute result with DSA version
    sort_insertion_dsa(input, calculated_output, N);

    // Compare results with tolerance
    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > 1e-5f) {
            printf("sort_insertion: FAILED\n");
            return 1;
        }
    }

    // Verify sortedness
    for (uint32_t i = 0; i < N - 1; i++) {
        if (expect_output[i] > expect_output[i + 1]) {
            printf("sort_insertion: FAILED\n");
            return 1;
        }
    }

    printf("sort_insertion: PASSED\n");
    return 0;
}

