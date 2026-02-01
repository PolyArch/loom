// Loom kernel implementation: interpolate_linear
#include "interpolate_linear.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Linear interpolation
// Tests complete compilation chain with search loop and floating-point arithmetic
// Test: 10 data points, 7 query points → interpolated results

// CPU implementation of linear interpolation
// Performs linear interpolation on N_query points
// input_x: known x coordinates (N_data elements, must be sorted)
// input_y: known y coordinates (N_data elements)
// input_xq: query x coordinates (N_query elements)
// output_yq: interpolated y coordinates (N_query elements)
// For query point xq, find interval [x[i], x[i+1]] containing xq
// and compute yq = y[i] + (xq - x[i]) * (y[i+1] - y[i]) / (x[i+1] - x[i])
void interpolate_linear_cpu(const float* __restrict__ input_x,
                             const float* __restrict__ input_y,
                             const float* __restrict__ input_xq,
                             float* __restrict__ output_yq,
                             const uint32_t N_data,
                             const uint32_t N_query) {
    for (uint32_t q = 0; q < N_query; q++) {
        float xq = input_xq[q];

        // Find the interval [x[i], x[i+1]] containing xq
        uint32_t i = 0;
        for (uint32_t k = 0; k < N_data - 1; k++) {
            if (xq >= input_x[k] && xq <= input_x[k + 1]) {
                i = k;
                break;
            }
        }

        // Linear interpolation
        float x0 = input_x[i];
        float x1 = input_x[i + 1];
        float y0 = input_y[i];
        float y1 = input_y[i + 1];

        float t = (xq - x0) / (x1 - x0);
        output_yq[q] = y0 + t * (y1 - y0);
    }
}

// Accelerator implementation of linear interpolation
LOOM_ACCEL()
void interpolate_linear_dsa(const float* __restrict__ input_x,
                             const float* __restrict__ input_y,
                             const float* __restrict__ input_xq,
                             float* __restrict__ output_yq,
                             const uint32_t N_data,
                             const uint32_t N_query) {
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t q = 0; q < N_query; q++) {
        float xq = input_xq[q];

        // Find the interval [x[i], x[i+1]] containing xq
        uint32_t i = 0;
        for (uint32_t k = 0; k < N_data - 1; k++) {
            if (xq >= input_x[k] && xq <= input_x[k + 1]) {
                i = k;
                break;
            }
        }

        // Linear interpolation
        float x0 = input_x[i];
        float x1 = input_x[i + 1];
        float y0 = input_y[i];
        float y1 = input_y[i + 1];

        float t = (xq - x0) / (x1 - x0);
        output_yq[q] = y0 + t * (y1 - y0);
    }
}

