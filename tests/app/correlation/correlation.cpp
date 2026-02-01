// Loom kernel implementation: correlation
#include "correlation.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Cross-correlation
// Tests complete compilation chain with nested loops and sliding window pattern (lag+i indexing)
// Test: x=[1,2,3,4,5,6], y=[1,0.5,0.25], sizes=(6,3) → output=[2.75,4.5,6.25,8]

// CPU implementation of cross-correlation
// Cross-correlation: corr[lag] = sum(x[i] * y[i+lag])
// Output size = x_size - y_size + 1
void correlation_cpu(const float* __restrict__ x,
                     const float* __restrict__ y,
                     float* __restrict__ output,
                     const uint32_t x_size,
                     const uint32_t y_size) {
    uint32_t output_size = x_size - y_size + 1;

    for (uint32_t lag = 0; lag < output_size; lag++) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < y_size; i++) {
            sum += x[lag + i] * y[i];
        }
        output[lag] = sum;
    }
}

// Cross-correlation: output[lag] = sum(x[lag+i] * y[i]) for i=0..y_size-1
// Accelerator implementation of cross-correlation
LOOM_ACCEL()
void correlation_dsa(const float* __restrict__ x,
                     const float* __restrict__ y,
                     float* __restrict__ output,
                     const uint32_t x_size,
                     const uint32_t y_size) {
    uint32_t output_size = x_size - y_size + 1;

    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t lag = 0; lag < output_size; lag++) {
        float sum = 0.0f;
        for (uint32_t i = 0; i < y_size; i++) {
            sum += x[lag + i] * y[i];
        }
        output[lag] = sum;
    }
}

