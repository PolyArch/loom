#include <cstdio>

#include "softmax.h"
#include <cmath>

int main() {
    const uint32_t N = 128;

    // Input array
    float input[N];

    // Output arrays
    float expect_output[N];
    float calculated_output[N];

    // Initialize input
    for (uint32_t i = 0; i < N; i++) {
        input[i] = static_cast<float>(i % 20) - 10.0f;
    }

    // Compute expected result with CPU version
    softmax_cpu(input, expect_output, N);

    // Compute result with accelerator version
    softmax_dsa(input, calculated_output, N);

    // Compare results
    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > 1e-5f) {
            printf("softmax: FAILED\n");
            return 1;
        }
    }

    // Verify sum is approximately 1.0
    float sum = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
        sum += expect_output[i];
    }
    if (fabsf(sum - 1.0f) > 1e-4f) {
        printf("softmax: FAILED\n");
        return 1;
    }

    printf("softmax: PASSED\n");
    return 0;
}

