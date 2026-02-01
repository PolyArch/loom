#include <cstdio>

#include "cumsum.h"
#include <cmath>

int main() {
    const uint32_t N = 1024;

    // Allocate and initialize input
    float input[N];
    for (uint32_t i = 0; i < N; i++) {
        input[i] = static_cast<float>(i % 10) + 1.0f;
    }

    // Allocate output arrays
    float expect_output[N];
    float calculated_output[N];

    // Compute expected result with CPU version
    cumsum_cpu(input, expect_output, N);

    // Compute result with DSA version
    cumsum_dsa(input, calculated_output, N);

    // Compare results with tolerance
    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > 1e-3f) {
            printf("cumsum: FAILED\n");
            return 1;
        }
    }

    printf("cumsum: PASSED\n");
    return 0;
}

