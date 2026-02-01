#include <cstdio>

#include "mmtile.h"

int main() {
    const uint32_t M = 64;
    const uint32_t N = 64;
    const uint32_t K = 64;
    
    // Tile sizes
    const uint32_t TILE_M = 8;
    const uint32_t TILE_N = 16;
    const uint32_t TILE_K = 8;
    
    // Input matrices
    uint32_t A[M * N];
    uint32_t B[N * K];
    
    // Output matrices
    uint32_t expect_C[M * K];
    uint32_t calculated_C[M * K];
    
    // Initialize input matrices
    for (uint32_t i = 0; i < M * N; i++) {
        A[i] = i % 10;
    }
    for (uint32_t i = 0; i < N * K; i++) {
        B[i] = (i + 1) % 10;
    }
    
    // Compute expected result with CPU version
    mmtile_cpu(A, B, expect_C, M, N, K, TILE_M, TILE_N, TILE_K);
    
    // Compute result with accelerator version
    mmtile_dsa(A, B, calculated_C, M, N, K, TILE_M, TILE_N, TILE_K);
    
    // Compare results
    for (uint32_t i = 0; i < M * K; i++) {
        if (expect_C[i] != calculated_C[i]) {
            printf("mmtile: FAILED\n");
            return 1;
        }
    }
    
    printf("mmtile: PASSED\n");
    return 0;
}

