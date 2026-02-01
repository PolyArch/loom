#include <cstdio>

#include "gauss_seidel_step.h"
#include <cmath>

int main() {
    const uint32_t N = 32;
    
    // Input matrix, RHS, and current solution
    float input_A[N * N];
    float input_b[N];
    float input_x[N];
    
    // Output solution vectors
    float expect_x[N];
    float calculated_x[N];
    
    // Initialize diagonally dominant matrix for stability
    for (uint32_t i = 0; i < N; i++) {
        for (uint32_t j = 0; j < N; j++) {
            if (i == j) {
                input_A[i * N + j] = 10.0f;
            } else {
                input_A[i * N + j] = 1.0f;
            }
        }
        input_b[i] = (float)(i + 1);
        input_x[i] = 0.0f;  // Initial guess
    }
    
    // Compute expected result with CPU version
    gauss_seidel_step_cpu(input_A, input_b, input_x, expect_x, N);
    
    // Compute result with accelerator version
    gauss_seidel_step_dsa(input_A, input_b, input_x, calculated_x, N);
    
    // Compare results with tolerance
    for (uint32_t i = 0; i < N; i++) {
        if (fabsf(expect_x[i] - calculated_x[i]) > 1e-5f) {
            printf("gauss_seidel_step: FAILED\n");
            return 1;
        }
    }
    
    printf("gauss_seidel_step: PASSED\n");
    return 0;
}

