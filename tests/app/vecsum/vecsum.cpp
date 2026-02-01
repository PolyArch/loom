// Loom kernel implementation: vecsum
#include "vecsum.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>



// Full pipeline test from C++ source: Vector sum (reduction)
// Tests complete compilation chain with reduction operation
// Then tests simulation: A=[0..15], init_value=0, N=16 → sum=120






// CPU implementation of vector sum
uint32_t vecsum_cpu(const uint32_t* __restrict__ A, 
                    const uint32_t init_value, 
                    const uint32_t N) {
    uint32_t sum = init_value;
    for (uint32_t i = 0; i < N; i++) {
        sum += A[i];
    }
    return sum;
}

// Vector sum: return sum(A[i])
// Accelerator implementation of vector sum
LOOM_TARGET("temporal")
LOOM_ACCEL()
uint32_t vecsum_dsa(const uint32_t* __restrict__ A, 
                    const uint32_t init_value, 
                    const uint32_t N) {
    LOOM_REDUCE(+)
    uint32_t sum = init_value;
    LOOM_PARALLEL(4)
    for (uint32_t i = 0; i < N; i++) {
        sum += A[i];
    }
    return sum;
}




