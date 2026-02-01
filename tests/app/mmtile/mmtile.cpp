// Loom kernel implementation: mmtile
#include "mmtile.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Tiled matrix multiplication with blocking
// Tests complete compilation chain with nested loops and tiling optimization
// Test: A(2x2) * B(2x2 identity) = [1,2,3,4] with tile sizes (1,1,1)

// CPU implementation of tiled matrix multiplication
// A: M x N matrix
// B: N x K matrix
// C: M x K matrix (output)
// TILE_M, TILE_N, TILE_K: tile sizes for blocking
void mmtile_cpu(const uint32_t* __restrict__ A, 
                const uint32_t* __restrict__ B, 
                uint32_t* __restrict__ C, 
                const uint32_t M, 
                const uint32_t N, 
                const uint32_t K,
                const uint32_t TILE_M,
                const uint32_t TILE_N,
                const uint32_t TILE_K) {
    // Initialize output matrix to zero
    for (uint32_t i = 0; i < M * K; i++) {
        C[i] = 0;
    }

    // Tiled matrix multiplication
    for (uint32_t i0 = 0; i0 < M; i0 += TILE_M) {
        for (uint32_t j0 = 0; j0 < K; j0 += TILE_K) {
            for (uint32_t k0 = 0; k0 < N; k0 += TILE_N) {
                // Compute the current tile
                uint32_t i_end = std::min(i0 + TILE_M, M);
                uint32_t j_end = std::min(j0 + TILE_K, K);
                uint32_t k_end = std::min(k0 + TILE_N, N);

                for (uint32_t i = i0; i < i_end; i++) {
                    for (uint32_t j = j0; j < j_end; j++) {
                        uint32_t sum = 0;
                        for (uint32_t k = k0; k < k_end; k++) {
                            sum += A[i * N + k] * B[k * K + j];
                        }
                        C[i * K + j] += sum;
                    }
                }
            }
        }
    }
}

// Accelerator implementation of tiled matrix multiplication
LOOM_ACCEL()
void mmtile_dsa(LOOM_MEMORY_BANK(4, block) LOOM_STREAM const uint32_t* __restrict__ A, 
                LOOM_STREAM const uint32_t* __restrict__ B, 
                uint32_t* __restrict__ C, 
                const uint32_t M, 
                const uint32_t N, 
                const uint32_t K,
                const uint32_t TILE_M,
                const uint32_t TILE_N,
                const uint32_t TILE_K) {
    // Initialize output matrix to zero
    for (uint32_t i = 0; i < M * K; i++) {
        C[i] = 0;
    }

    // Tiled matrix multiplication
    LOOM_PARALLEL(4, contiguous)
    for (uint32_t i0 = 0; i0 < M; i0 += TILE_M) {
        LOOM_UNROLL(4)
        for (uint32_t j0 = 0; j0 < K; j0 += TILE_K) {
            LOOM_TRIPCOUNT_FULL(16, 16, 1, 64)
            for (uint32_t k0 = 0; k0 < N; k0 += TILE_N) {
                // Compute the current tile
                uint32_t i_end = std::min(i0 + TILE_M, M);
                uint32_t j_end = std::min(j0 + TILE_K, K);
                uint32_t k_end = std::min(k0 + TILE_N, N);

                for (uint32_t i = i0; i < i_end; i++) {
                    for (uint32_t j = j0; j < j_end; j++) {
                        uint32_t sum = 0;
                        for (uint32_t k = k0; k < k_end; k++) {
                            sum += A[i * N + k] * B[k * K + j];
                        }
                        C[i * K + j] += sum;
                    }
                }
            }
        }
    }
}

