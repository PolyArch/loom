// Loom kernel implementation: matvec
#include "matvec.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Matrix-vector multiplication
// Tests complete compilation chain with nested loops and reduction
// Test: (2x3)*[1,2,3] = [14,32]






// CPU implementation of matrix-vector multiplication
// A: M x N matrix (row-major)
// x: vector of length N
// y: vector of length M (output)
// y = A * x
void matvec_cpu(const uint32_t* __restrict__ A, 
                const uint32_t* __restrict__ x, 
                uint32_t* __restrict__ y, 
                const uint32_t M, 
                const uint32_t N) {
    for (uint32_t i = 0; i < M; i++) {
        uint32_t sum = 0;
        for (uint32_t j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}

// Matrix-vector multiplication: y[i] = sum_j A[i,j] * x[j]
// Accelerator implementation of matrix-vector multiplication
LOOM_ACCEL()
void matvec_dsa(LOOM_MEMORY_BANK(4, block) LOOM_STREAM const uint32_t* __restrict__ A, 
                LOOM_STREAM const uint32_t* __restrict__ x, 
                uint32_t* __restrict__ y, 
                const uint32_t M, 
                const uint32_t N) {
    for (uint32_t i = 0; i < M; i++) {
        uint32_t sum = 0;
        for (uint32_t j = 0; j < N; j++) {
            sum += A[i * N + j] * x[j];
        }
        y[i] = sum;
    }
}





