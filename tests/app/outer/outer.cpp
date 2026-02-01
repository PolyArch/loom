// Loom kernel implementation: outer
#include "outer.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Outer product
// Tests complete compilation chain with nested loops
// Test: a=[1,2], b=[3,4] → C=[3,4,6,8]






// CPU implementation of outer product
// a: vector of length M
// b: vector of length N
// C: M x N matrix (output, row-major)
// C[i,j] = a[i] * b[j]
void outer_cpu(const uint32_t* __restrict__ a, 
               const uint32_t* __restrict__ b, 
               uint32_t* __restrict__ C, 
               const uint32_t M, 
               const uint32_t N) {
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            C[i * N + j] = a[i] * b[j];
        }
    }
}

// Outer product: C[i,j] = a[i] * b[j]
// Accelerator implementation of outer product
LOOM_ACCEL()
void outer_dsa(const uint32_t* __restrict__ a, 
               const uint32_t* __restrict__ b, 
               uint32_t* __restrict__ C, 
               const uint32_t M, 
               const uint32_t N) {
    LOOM_PARALLEL(4)
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            C[i * N + j] = a[i] * b[j];
        }
    }
}


// Read vector a (M elements)
// Read vector b (N elements)
// Repeat each element of a N times: [a0, a0, ..., a0, a1, a1, ..., a1, ...]
// Repeat entire b vector M times: [b0, b1, ..., bN, b0, b1, ..., bN, ...]
// Element-wise multiply to get outer product
// Write result to C (M*N elements)


