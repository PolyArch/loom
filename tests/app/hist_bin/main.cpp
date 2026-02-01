#include <cstdio>

#include "hist_bin.h"

int main() {
    const uint32_t N = 1024;
    const uint32_t num_bins = 10;
    const float min_val = 0.0f;
    const float max_val = 100.0f;

    // Allocate and initialize input
    float input[N];
    for (uint32_t i = 0; i < N; i++) {
        input[i] = static_cast<float>(i % 100);
    }

    // Allocate output arrays
    uint32_t expect_output[num_bins];
    uint32_t calculated_output[num_bins];

    // Compute expected result with CPU version
    hist_bin_cpu(input, expect_output, N, num_bins, min_val, max_val);

    // Compute result with DSA version
    hist_bin_dsa(input, calculated_output, N, num_bins, min_val, max_val);

    // Compare results
    for (uint32_t i = 0; i < num_bins; i++) {
        if (expect_output[i] != calculated_output[i]) {
            printf("hist_bin: FAILED\n");
            return 1;
        }
    }

    printf("hist_bin: PASSED\n");
    return 0;
}

