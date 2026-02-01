// Loom kernel implementation: vecnorm_l2
#include "vecnorm_l2.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Vector L2 norm squared
// Tests complete compilation chain with multiplication and reduction
// Test: L2 norm squared of [8, 5, 12, 3, 9, 6, 15, 4, 11, 7] = 770
// (Actual L2 norm = sqrt(770) ≈ 27.75)

// CPU implementation of vector L2 norm (squared, to avoid floating point)
uint32_t vecnorm_l2_cpu(const uint32_t* __restrict__ A, 
                        const uint32_t N) {
    uint32_t norm_sq = 0;
    for (uint32_t i = 0; i < N; i++) {
        norm_sq += A[i] * A[i];
    }
    return norm_sq;
}

// L2 norm squared: return sum(A[i] * A[i])
// Accelerator implementation of vector L2 norm (squared)
LOOM_ACCEL()
uint32_t vecnorm_l2_dsa(const uint32_t* __restrict__ A, 
                        const uint32_t N) {
    uint32_t norm_sq = 0;
    LOOM_PARALLEL(4, contiguous)
    LOOM_TRIPCOUNT_FULL(256, 256, 1, 1024)
    for (uint32_t i = 0; i < N; i++) {
        norm_sq += A[i] * A[i];
    }
    return norm_sq;
}

// Step 1: Read vector A

// Step 2: Duplicate A to get two identical dataflows for A[i] * A[i]

// Step 3: Element-wise multiply A[i] * A[i]

// Step 4: Sum all squared values

