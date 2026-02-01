// Loom kernel implementation: jacobi_stencil_7pt
#include "jacobi_stencil_7pt.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: 7-point Jacobi stencil (3D)
// Tests complete compilation chain with 3D stencil computation
// Test: 4x4x4 grid with 7-point stencil → interior points updated, boundary unchanged







// CPU implementation of 7-point Jacobi stencil (3D grid)
// Updates interior points of a 3D grid using 7-point stencil
// input_grid: input grid (L x M x N elements, indexed as [k*M*N + i*N + j])
// output_grid: output grid (L x M x N elements)
// Interior points: output[k,i,j] = (1/6) * sum of 6 neighbors
// Boundary points are copied unchanged
void jacobi_stencil_7pt_cpu(const float* __restrict__ input_grid,
                             float* __restrict__ output_grid,
                             const uint32_t L,
                             const uint32_t M,
                             const uint32_t N) {
    const uint32_t MN = M * N;
    
    // Copy all boundary points
    for (uint32_t k = 0; k < L; k++) {
        for (uint32_t i = 0; i < M; i++) {
            for (uint32_t j = 0; j < N; j++) {
                uint32_t idx = k * MN + i * N + j;
                // Boundary condition: any edge of the 3D grid
                if (k == 0 || k == L - 1 || i == 0 || i == M - 1 || j == 0 || j == N - 1) {
                    output_grid[idx] = input_grid[idx];
                }
            }
        }
    }
    
    // Update interior points with 7-point stencil
    for (uint32_t k = 1; k < L - 1; k++) {
        for (uint32_t i = 1; i < M - 1; i++) {
            for (uint32_t j = 1; j < N - 1; j++) {
                float front = input_grid[(k - 1) * MN + i * N + j];
                float back = input_grid[(k + 1) * MN + i * N + j];
                float north = input_grid[k * MN + (i - 1) * N + j];
                float south = input_grid[k * MN + (i + 1) * N + j];
                float west = input_grid[k * MN + i * N + (j - 1)];
                float east = input_grid[k * MN + i * N + (j + 1)];
                output_grid[k * MN + i * N + j] = (1.0f / 6.0f) * (front + back + north + south + west + east);
                // printf("output[%u]=%e\n", k * MN + i * N + j, output_grid[k * MN + i * N + j]);
            }
        }
    }
    // // Print output_imag
    // for (uint32_t i = 0; i < 64; i++) {
    //     printf("%e", output_grid[i]);
    //     if (i < 63) printf(",");
    // }
    // printf(" \n");
}

// Accelerator implementation of 7-point Jacobi stencil
LOOM_ACCEL()
void jacobi_stencil_7pt_dsa(const float* __restrict__ input_grid,
                             float* __restrict__ output_grid,
                             const uint32_t L,
                             const uint32_t M,
                             const uint32_t N) {
    const uint32_t MN = M * N;
    
    // Copy all boundary points
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t k = 0; k < L; k++) {
        for (uint32_t i = 0; i < M; i++) {
            for (uint32_t j = 0; j < N; j++) {
                uint32_t idx = k * MN + i * N + j;
                if (k == 0 || k == L - 1 || i == 0 || i == M - 1 || j == 0 || j == N - 1) {
                    output_grid[idx] = input_grid[idx];
                }
            }
        }
    }
    
    // Update interior points with 7-point stencil
    for (uint32_t k = 1; k < L - 1; k++) {
        for (uint32_t i = 1; i < M - 1; i++) {
            for (uint32_t j = 1; j < N - 1; j++) {
                float front = input_grid[(k - 1) * MN + i * N + j];
                float back = input_grid[(k + 1) * MN + i * N + j];
                float north = input_grid[k * MN + (i - 1) * N + j];
                float south = input_grid[k * MN + (i + 1) * N + j];
                float west = input_grid[k * MN + i * N + (j - 1)];
                float east = input_grid[k * MN + i * N + (j + 1)];
                output_grid[k * MN + i * N + j] = (1.0f / 6.0f) * (front + back + north + south + west + east);
            }
        }
    }
}



