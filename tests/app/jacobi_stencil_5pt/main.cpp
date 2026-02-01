#include <cstdio>

#include "jacobi_stencil_5pt.h"
#include <cmath>

int main() {
    const uint32_t M = 32;  // Rows
    const uint32_t N = 32;  // Columns
    
    // Input and output grids
    float input_grid[M * N];
    float expect_grid[M * N];
    float calculated_grid[M * N];
    
    // Initialize input grid
    for (uint32_t i = 0; i < M; i++) {
        for (uint32_t j = 0; j < N; j++) {
            input_grid[i * N + j] = (float)(i + j);
        }
    }
    
    // Compute expected result with CPU version
    jacobi_stencil_5pt_cpu(input_grid, expect_grid, M, N);
    
    // Compute result with accelerator version
    jacobi_stencil_5pt_dsa(input_grid, calculated_grid, M, N);
    
    // Compare results with tolerance
    for (uint32_t i = 0; i < M * N; i++) {
        if (fabsf(expect_grid[i] - calculated_grid[i]) > 1e-5f) {
            printf("jacobi_stencil_5pt: FAILED\n");
            return 1;
        }
    }
    
    printf("jacobi_stencil_5pt: PASSED\n");
    return 0;
}

