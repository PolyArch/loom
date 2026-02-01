// Loom kernel implementation: variance
#include "variance.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Variance computation (two-pass algorithm)
// Tests complete compilation chain with multiple reduction operations
// Test: Compute variance of [12.5, 8.3, 15.7, 6.2, 11, 9.8, 13.4, 7.1, 14.6, 10.9]
// Mean = 10.95, Variance = 9.0025






// CPU implementation of variance computation
// variance = sum((x_i - mean)^2) / N
float variance_cpu(const float* __restrict__ input,
                   const uint32_t N) {
    // Compute mean
    float sum = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
        sum += input[i];
    }
    float mean = sum / static_cast<float>(N);
    
    // Compute variance
    float var = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
        float diff = input[i] - mean;
        var += diff * diff;
    }
    return var / static_cast<float>(N);
}

// Variance: return sum((input[i] - mean)^2) / N
// Accelerator implementation of variance computation
LOOM_TARGET("temporal")
LOOM_ACCEL()
float variance_dsa(const float* __restrict__ input,
                   const uint32_t N) {
    // Compute mean
    LOOM_REDUCE(+)
    float sum = 0.0f;
    LOOM_PARALLEL(4)
    for (uint32_t i = 0; i < N; i++) {
        sum += input[i];
    }
    float mean = sum / static_cast<float>(N);
    
    // Compute variance
    float var = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
        float diff = input[i] - mean;
        var += diff * diff;
    }
    return var / static_cast<float>(N);
}



// === First pass: compute mean ===

// === Second pass: compute variance ===



