// Loom kernel implementation: softmax
#include "softmax.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Softmax activation function
// Tests complete compilation chain with multi-pass algorithm (max reduction, exp, sum, normalize)
// Test: [1.0, 2.0, 3.0] → [0.090, 0.245, 0.665] (numerically stable softmax)

// CPU implementation of Softmax activation
// softmax(x_i) = exp(x_i) / sum(exp(x_j))
void softmax_cpu(const float* __restrict__ input,
                 float* __restrict__ output,
                 const uint32_t N) {
    // Find max for numerical stability
    float max_val = input[0];
    for (uint32_t i = 1; i < N; i++) {
        max_val = std::max(max_val, input[i]);
    }

    // Compute exp(x - max) and sum
    float sum = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    // Normalize
    for (uint32_t i = 0; i < N; i++) {
        output[i] = output[i] / sum;
    }
}

// Softmax: output[i] = exp(input[i] - max) / sum(exp(input[j] - max))
// Accelerator implementation of Softmax activation
LOOM_TARGET("temporal")
LOOM_ACCEL()
void softmax_dsa(const float* __restrict__ input,
                 float* __restrict__ output,
                 const uint32_t N) {
    // Find max for numerical stability
    float max_val = input[0];
    LOOM_PARALLEL(4)
    for (uint32_t i = 1; i < N; i++) {
        max_val = std::max(max_val, input[i]);
    }

    // Compute exp(x - max) and sum
    LOOM_REDUCE(+)
    float sum = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
        output[i] = expf(input[i] - max_val);
        sum += output[i];
    }

    // Normalize
    for (uint32_t i = 0; i < N; i++) {
        output[i] = output[i] / sum;
    }
}

