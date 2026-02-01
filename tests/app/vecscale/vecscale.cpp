// Loom kernel implementation: vecscale
#include "vecscale.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>



// Full pipeline test from C++ source: Vector scaling (scalar-vector multiplication)
// Tests complete compilation chain with scalar-vector operation





// CPU implementation of vector scaling (scalar × vector)
void vecscale_cpu(const uint32_t* __restrict__ A, 
                  const uint32_t alpha, 
                  uint32_t* __restrict__ B, 
                  const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        B[i] = alpha * A[i];
    }
}

// Vector scale: B[i] = alpha * A[i]
// Accelerator implementation of vector scaling
LOOM_ACCEL()
void vecscale_dsa(const uint32_t* __restrict__ A, 
                  const uint32_t alpha, 
                  uint32_t* __restrict__ B, 
                  const uint32_t N) {
    LOOM_PARALLEL(4, contiguous)
    LOOM_TRIPCOUNT_FULL(256, 256, 1, 1024)
    for (uint32_t i = 0; i < N; i++) {
        B[i] = alpha * A[i];
    }
}





