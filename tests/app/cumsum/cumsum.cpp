// Loom kernel implementation: cumsum
#include "cumsum.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Cumulative sum (prefix sum, inclusive scan)
// Tests complete compilation chain with scan operation
// Test: input=[1,2,3,4,5], N=5 → output=[1,3,6,10,15]






// CPU implementation of cumulative sum
// output[i] = sum of input[0..i]
void cumsum_cpu(const float* __restrict__ input,
                float* __restrict__ output,
                const uint32_t N) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < N; i++) {
        sum += input[i];
        output[i] = sum;
    }
}

// Cumulative sum: output[i] = sum(input[0..i])
// Accelerator implementation of cumulative sum
LOOM_TARGET("temporal")
LOOM_ACCEL()
void cumsum_dsa(const float* __restrict__ input,
                float* __restrict__ output,
                const uint32_t N) {
    LOOM_REDUCE(+)
    float sum = 0.0f;
    LOOM_PARALLEL(4)
    for (uint32_t i = 0; i < N; i++) {
        sum += input[i];
        output[i] = sum;
    }
}




