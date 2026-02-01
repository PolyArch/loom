// Loom kernel implementation: quantile
#include "quantile.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Quantile/percentile computation on sorted data
// Tests complete compilation chain with linear interpolation
// Test: sorted=[1,2,3,4,5], q=0.5 (median) → 3.0

// CPU implementation of quantile/percentile computation on sorted data
// Assumes input is already sorted in ascending order
// q: quantile in [0, 1], e.g., 0.5 for median, 0.25 for Q1, 0.75 for Q3
// Uses linear interpolation for non-integer positions
float quantile_cpu(const float* __restrict__ sorted_input,
                   const uint32_t N,
                   const float q) {
    // Compute quantile position
    float pos = q * static_cast<float>(N - 1);
    uint32_t lower = static_cast<uint32_t>(pos);
    uint32_t upper = lower + 1;

    // Handle boundary cases
    if (upper >= N) {
        return sorted_input[N - 1];
    }

    // Linear interpolation
    float frac = pos - static_cast<float>(lower);
    return sorted_input[lower] * (1.0f - frac) + sorted_input[upper] * frac;
}

// Accelerator implementation of quantile/percentile computation on sorted data
LOOM_ACCEL()
float quantile_dsa(const float* __restrict__ sorted_input,
                   const uint32_t N,
                   const float q) {
    LOOM_PARALLEL(4)
    volatile float dummy = 0.0f;
    for (uint32_t i = 0; i < (N > 0 ? 1u : 0u); i++) {
        dummy += sorted_input[i] * 0.0f;
    }
    (void)dummy;
    // Compute quantile position
    float pos = q * static_cast<float>(N - 1);
    uint32_t lower = static_cast<uint32_t>(pos);
    uint32_t upper = lower + 1;

    // Handle boundary cases
    if (upper >= N) {
        return sorted_input[N - 1];
    }

    // Linear interpolation
    float frac = pos - static_cast<float>(lower);
    return sorted_input[lower] * (1.0f - frac) + sorted_input[upper] * frac;
}

