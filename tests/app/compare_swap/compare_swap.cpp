// Loom kernel implementation: compare_swap
#include "compare_swap.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Compare and swap operation
// Tests complete compilation chain with conditional assignment (output min/max of two values)
// Test: a=[3,1,5,2], b=[2,4,3,6], N=4 → min=[2,1,3,2], max=[3,4,5,6]

// CPU implementation of compare and swap
void compare_swap_cpu(const float* __restrict__ input_a,
                      const float* __restrict__ input_b,
                      float* __restrict__ output_min,
                      float* __restrict__ output_max,
                      const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        if (input_a[i] <= input_b[i]) {
            output_min[i] = input_a[i];
            output_max[i] = input_b[i];
        } else {
            output_min[i] = input_b[i];
            output_max[i] = input_a[i];
        }
    }
}

// Compare-swap: output_min[i] = min(a, b); output_max[i] = max(a, b)
// Accelerator implementation of compare and swap
LOOM_ACCEL()
void compare_swap_dsa(const float* __restrict__ input_a,
                      const float* __restrict__ input_b,
                      float* __restrict__ output_min,
                      float* __restrict__ output_max,
                      const uint32_t N) {
    LOOM_PARALLEL(4, interleaved)
    LOOM_TRIPCOUNT_RANGE(10, 1000)
    for (uint32_t i = 0; i < N; i++) {
        if (input_a[i] <= input_b[i]) {
            output_min[i] = input_a[i];
            output_max[i] = input_b[i];
        } else {
            output_min[i] = input_b[i];
            output_max[i] = input_a[i];
        }
    }
}

