// Loom kernel implementation: prefix_sum_exclusive
#include "prefix_sum_exclusive.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Exclusive prefix sum (scan)
// Tests complete compilation chain with scan operation
// Test: [1,2,3,4] → [0,1,3,6] (output[i] = sum of input[0..i-1])






// CPU implementation of exclusive prefix sum
// output[0] = 0, output[i] = sum(input[0:i]) for i > 0
void prefix_sum_exclusive_cpu(const uint32_t* __restrict__ input,
                              uint32_t* __restrict__ output,
                              const uint32_t N) {
    if (N == 0) return;
    
    output[0] = 0;
    for (uint32_t i = 1; i < N; i++) {
        output[i] = output[i - 1] + input[i - 1];
    }
}

// Prefix sum (exclusive): output[0] = 0, output[i] = sum(input[0..i-1])
// Accelerator implementation of exclusive prefix sum
LOOM_TARGET("temporal")
LOOM_ACCEL()
void prefix_sum_exclusive_dsa(const uint32_t* __restrict__ input,
                              LOOM_REDUCE(+) uint32_t* __restrict__ output,
                              const uint32_t N) {
    if (N == 0) return;
    
    output[0] = 0;
    LOOM_PARALLEL(4)
    for (uint32_t i = 1; i < N; i++) {
        output[i] = output[i - 1] + input[i - 1];
    }
}




