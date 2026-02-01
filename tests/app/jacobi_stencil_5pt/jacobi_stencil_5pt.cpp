// Loom kernel implementation: jacobi_stencil_5pt
#include "jacobi_stencil_5pt.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: 5-point Jacobi stencil (2D)
// Tests complete compilation chain with stencil computation
// Test: 4x4 grid with 5-point stencil averaging → interior updated, boundary unchanged






// CPU implementation of 5-point Jacobi stencil (2D grid)
// Updates interior points of a 2D grid using 5-point stencil
// input_grid: input grid (M x N elements, row-major)
// output_grid: output grid (M x N elements, row-major)
// Interior points: output[i,j] = 0.25 * (input[i-1,j] + input[i+1,j] + input[i,j-1] + input[i,j+1])
// Boundary points are copied unchanged
void jacobi_stencil_5pt_cpu(const float* __restrict__ input_grid,
                             float* __restrict__ output_grid,
                             const uint32_t M,
                             const uint32_t N) {
    // Copy boundary points
    for (uint32_t j = 0; j < N; j++) {
        output_grid[0 * N + j] = input_grid[0 * N + j];  // Top row
        output_grid[(M - 1) * N + j] = input_grid[(M - 1) * N + j];  // Bottom row
    }
    for (uint32_t i = 0; i < M; i++) {
        output_grid[i * N + 0] = input_grid[i * N + 0];  // Left column
        output_grid[i * N + (N - 1)] = input_grid[i * N + (N - 1)];  // Right column
    }
    
    // Update interior points with 5-point stencil
    for (uint32_t i = 1; i < M - 1; i++) {
        for (uint32_t j = 1; j < N - 1; j++) {
            float center = input_grid[i * N + j];
            float north = input_grid[(i - 1) * N + j];
            float south = input_grid[(i + 1) * N + j];
            float west = input_grid[i * N + (j - 1)];
            float east = input_grid[i * N + (j + 1)];
            output_grid[i * N + j] = 0.25f * (north + south + west + east);
        }
    }
}

// Accelerator implementation of 5-point Jacobi stencil
LOOM_ACCEL()
void jacobi_stencil_5pt_dsa(const float* __restrict__ input_grid,
                             float* __restrict__ output_grid,
                             const uint32_t M,
                             const uint32_t N) {
    // Copy boundary points
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t j = 0; j < N; j++) {
        output_grid[0 * N + j] = input_grid[0 * N + j];
        output_grid[(M - 1) * N + j] = input_grid[(M - 1) * N + j];
    }
    for (uint32_t i = 0; i < M; i++) {
        output_grid[i * N + 0] = input_grid[i * N + 0];
        output_grid[i * N + (N - 1)] = input_grid[i * N + (N - 1)];
    }
    
    // Update interior points with 5-point stencil
    for (uint32_t i = 1; i < M - 1; i++) {
        for (uint32_t j = 1; j < N - 1; j++) {
            float center = input_grid[i * N + j];
            float north = input_grid[(i - 1) * N + j];
            float south = input_grid[(i + 1) * N + j];
            float west = input_grid[i * N + (j - 1)];
            float east = input_grid[i * N + (j + 1)];
            output_grid[i * N + j] = 0.25f * (north + south + west + east);
        }
    }
}



