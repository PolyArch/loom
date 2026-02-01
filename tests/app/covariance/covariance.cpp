// Loom kernel implementation: covariance
#include "covariance.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Covariance between two vectors
// Tests complete compilation chain with multi-pass algorithm (compute means, then covariance) and scalar return
// Test: X=[1,2,3,4,5], Y=[2,4,5,4,5], N=5 → return 1.2

// CPU implementation of covariance between two vectors
// cov(X,Y) = sum((x_i - mean_x) * (y_i - mean_y)) / N
float covariance_cpu(const float* __restrict__ X,
                     const float* __restrict__ Y,
                     const uint32_t N) {
    // Compute means
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
        sum_x += X[i];
        sum_y += Y[i];
    }
    float mean_x = sum_x / static_cast<float>(N);
    float mean_y = sum_y / static_cast<float>(N);

    // Compute covariance
    float cov = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
        cov += (X[i] - mean_x) * (Y[i] - mean_y);
    }
    return cov / static_cast<float>(N);
}

// Covariance: return sum((X[i] - mean_X) * (Y[i] - mean_Y)) / N
// Accelerator implementation of covariance between two vectors
LOOM_ACCEL()
float covariance_dsa(const float* __restrict__ X,
                     const float* __restrict__ Y,
                     const uint32_t N) {
    // Compute means
    float sum_x = 0.0f;
    float sum_y = 0.0f;
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t i = 0; i < N; i++) {
        sum_x += X[i];
        sum_y += Y[i];
    }
    float mean_x = sum_x / static_cast<float>(N);
    float mean_y = sum_y / static_cast<float>(N);

    // Compute covariance
    float cov = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
        cov += (X[i] - mean_x) * (Y[i] - mean_y);
    }
    return cov / static_cast<float>(N);
}

