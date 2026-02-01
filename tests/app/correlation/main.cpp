#include <cstdio>

#include "correlation.h"
#include <cmath>

int main() {
    const uint32_t x_size = 128;
    const uint32_t y_size = 16;
    const uint32_t output_size = x_size - y_size + 1;

    // Input signals
    float x[x_size];
    float y[y_size];

    // Output arrays for cross-correlation
    float expect_corr[output_size];
    float calculated_corr[output_size];

    // Initialize signals
    for (uint32_t i = 0; i < x_size; i++) {
        x[i] = sinf(2.0f * 3.14159f * i / 16.0f);
    }
    for (uint32_t i = 0; i < y_size; i++) {
        y[i] = cosf(2.0f * 3.14159f * i / 8.0f);
    }

    // Test cross-correlation
    correlation_cpu(x, y, expect_corr, x_size, y_size);
    correlation_dsa(x, y, calculated_corr, x_size, y_size);

    for (uint32_t i = 0; i < output_size; i++) {
        if (fabsf(expect_corr[i] - calculated_corr[i]) > 1e-4f) {
            printf("correlation: FAILED\n");
            return 1;
        }
    }

    printf("correlation: PASSED\n");
    return 0;
}

