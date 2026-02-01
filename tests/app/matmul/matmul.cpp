// Loom kernel implementation: matmul
#include "matmul.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Matrix multiplication
// Tests complete compilation chain with triple-nested loops
// Test: (2x3)*(3x2) = (2x2) → [22,28,49,64]

// CPU implementation of matrix multiplication
// A: M x N matrix
// B: N x K matrix
// C: M x K matrix (output)
void matmul_cpu(const uint32_t* __restrict__ A, 
                const uint32_t* __restrict__ B, 
                uint32_t* __restrict__ C, 
                const uint32_t M, 
                const uint32_t N, 
                const uint32_t K) {
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < K; j++) {
            uint32_t sum = 0;
            for (uint32_t k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

// Matrix multiplication: C[i,j] = sum_k A[i,k] * B[k,j]
// Accelerator implementation of matrix multiplication
LOOM_ACCEL()
void matmul_dsa(LOOM_MEMORY_BANK(4, block) LOOM_STREAM const uint32_t* __restrict__ A, 
                LOOM_STREAM const uint32_t* __restrict__ B, 
                uint32_t* __restrict__ C, 
                const uint32_t M, 
                const uint32_t N, 
                const uint32_t K) {
    LOOM_PARALLEL(4, contiguous)
    for (uint32_t i = 0; i < M; i++) {
        LOOM_UNROLL(4)
        for (uint32_t j = 0; j < K; j++) {
            uint32_t sum = 0;
            LOOM_TRIPCOUNT_FULL(16, 16, 1, 64)
            for (uint32_t k = 0; k < N; k++) {
                sum += A[i * N + k] * B[k * K + j];
            }
            C[i * K + j] = sum;
        }
    }
}

