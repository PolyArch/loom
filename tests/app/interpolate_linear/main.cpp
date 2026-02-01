#include <cstdio>

#include "interpolate_linear.h"
#include <cmath>

int main() {
    const uint32_t N_data = 32;
    const uint32_t N_query = 64;

    // Input data points
    float input_x[N_data];
    float input_y[N_data];

    // Query points
    float input_xq[N_query];

    // Output interpolated values
    float expect_yq[N_query];
    float calculated_yq[N_query];

    // Initialize data points (sorted x values)
    for (uint32_t i = 0; i < N_data; i++) {
        input_x[i] = (float)i;
        input_y[i] = (float)(i * i);  // y = x^2
    }

    // Initialize query points
    for (uint32_t i = 0; i < N_query; i++) {
        input_xq[i] = (float)i * 0.5f;  // Query at half-step intervals
    }

    // Compute expected result with CPU version
    interpolate_linear_cpu(input_x, input_y, input_xq, expect_yq, N_data, N_query);

    // Compute result with accelerator version
    interpolate_linear_dsa(input_x, input_y, input_xq, calculated_yq, N_data, N_query);

    // Compare results with tolerance
    for (uint32_t i = 0; i < N_query; i++) {
        if (fabsf(expect_yq[i] - calculated_yq[i]) > 1e-4f) {
            printf("interpolate_linear: FAILED\n");
            return 1;
        }
    }

    printf("interpolate_linear: PASSED\n");
    return 0;
}

