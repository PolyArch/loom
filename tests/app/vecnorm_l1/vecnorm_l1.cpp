// Loom kernel implementation: vecnorm_l1
#include "vecnorm_l1.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Vector L1 norm
// Tests complete compilation chain with sum reduction
// Test: L1 norm of [15, 7, 22, 9, 18, 4, 31, 12, 26, 13] = 157






// CPU implementation of vector L1 norm
uint32_t vecnorm_l1_cpu(const uint32_t* __restrict__ A, 
                        const uint32_t N) {
    uint32_t norm = 0;
    for (uint32_t i = 0; i < N; i++) {
        norm += A[i];
    }
    return norm;
}

// L1 norm: return sum(|A[i]|) = sum(A[i]) for unsigned
// Accelerator implementation of vector L1 norm
LOOM_ACCEL()
uint32_t vecnorm_l1_dsa(const uint32_t* __restrict__ A, 
                        const uint32_t N) {
    uint32_t norm = 0;
    LOOM_PARALLEL(4, contiguous)
    LOOM_TRIPCOUNT_FULL(256, 256, 1, 1024)
    for (uint32_t i = 0; i < N; i++) {
        norm += A[i];
    }
    return norm;
}





