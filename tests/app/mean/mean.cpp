// Loom kernel implementation: mean
#include "mean.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Mean computation (sum reduction + division)
// Tests complete compilation chain with reduction operation and post-processing
// Test: input=[1,2,3,4,5] → mean=3.0

// CPU implementation of mean computation
float mean_cpu(const float* __restrict__ input,
               const uint32_t N) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
        sum += input[i];
    }
    return sum / static_cast<float>(N);
}

// Mean: return sum(input[i]) / N
// Accelerator implementation of mean computation
LOOM_TARGET("temporal")
LOOM_ACCEL()
float mean_dsa(const float* __restrict__ input,
               const uint32_t N) {
    LOOM_REDUCE(+)
    float sum = 0.0f;
    LOOM_PARALLEL(4)
    for (uint32_t i = 0; i < N; i++) {
        sum += input[i];
    }
    return sum / static_cast<float>(N);
}

