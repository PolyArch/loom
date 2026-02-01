// Loom kernel implementation: transpose
#include "transpose.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Matrix transpose
// Tests complete compilation chain with nested loops and non-sequential memory access

// CPU implementation of matrix transpose
// A: M x N matrix (row-major)
// B: N x M matrix (row-major, output)
void transpose_cpu(const uint32_t* __restrict__ A, 
                   uint32_t* __restrict__ B, 
                   const uint32_t M, 
                   const uint32_t N) {
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            B[j * M + i] = A[i * N + j];
        }
    }
}

// Transpose: B[j * M + i] = A[i * N + j]
// Accelerator implementation of matrix transpose
LOOM_ACCEL()
void transpose_dsa(LOOM_MEMORY_BANK(8) LOOM_STREAM const uint32_t* __restrict__ A, 
                   LOOM_STREAM uint32_t* __restrict__ B, 
                   const uint32_t M, 
                   const uint32_t N) {
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            B[j * M + i] = A[i * N + j];
        }
    }
}

// For each row i of A, read linearly and write as column i of B (strided)
// Write to column i of B using 2D strided write (stride = M)

