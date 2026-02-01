// Loom kernel implementation: prefix_sum_inclusive
#include "prefix_sum_inclusive.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Inclusive prefix sum (scan)
// Tests complete compilation chain with scan operation
// Test: [1,2,3,4] → [1,3,6,10] (output[i] = sum of input[0..i])

// CPU implementation of prefix sum (inclusive scan)
// output[i] = sum(input[0:i+1])
void prefix_sum_inclusive_cpu(const uint32_t* __restrict__ input,
                              uint32_t* __restrict__ output,
                              const uint32_t N) {
    if (N == 0) return;

    output[0] = input[0];
    for (uint32_t i = 1; i < N; i++) {
        output[i] = output[i - 1] + input[i];
    }
}

// Prefix sum (inclusive): output[i] = sum(input[0..i])
// Accelerator implementation of prefix sum (inclusive scan)
LOOM_TARGET("temporal")
LOOM_ACCEL()
void prefix_sum_inclusive_dsa(const uint32_t* __restrict__ input,
                              LOOM_REDUCE(+) uint32_t* __restrict__ output,
                              const uint32_t N) {
    if (N == 0) return;

    output[0] = input[0];
    LOOM_PARALLEL(4)
    for (uint32_t i = 1; i < N; i++) {
        output[i] = output[i - 1] + input[i];
    }
}

