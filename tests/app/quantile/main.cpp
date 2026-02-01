#include <cstdio>

#include "quantile.h"
#include <cmath>

int main() {
    const uint32_t N = 1024;

    // Allocate and initialize already-sorted input
    float input[N];
    for (uint32_t i = 0; i < N; i++) {
        input[i] = static_cast<float>(i);  // Already sorted: 0, 1, 2, ..., N-1
    }

    // Test median (q = 0.5)
    float q = 0.5f;

    // Compute expected result with CPU version
    float expect_result = quantile_cpu(input, N, q);

    // Compute result with DSA version
    float calculated_result = quantile_dsa(input, N, q);

    // Compare results with tolerance
    if (fabsf(expect_result - calculated_result) > 1e-5f) {
        printf("quantile: FAILED\n");
        return 1;
    }

    // Verify expected median value
    float expected_median = 511.5f;  // (N-1) * 0.5 = 1023 * 0.5
    if (fabsf(expect_result - expected_median) > 1e-3f) {
        printf("quantile: FAILED\n");
        return 1;
    }

    printf("quantile: PASSED\n");
    return 0;
}

