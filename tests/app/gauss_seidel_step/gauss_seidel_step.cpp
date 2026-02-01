// Loom kernel implementation: gauss_seidel_step
#include "gauss_seidel_step.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Gauss-Seidel iteration step
// Tests complete compilation chain with loop-carried dependencies
// Test: A=[[2,1],[1,2]], b=[3,3], x=[0,0] → updated x=[1.5,0.75]






// CPU implementation of single Gauss-Seidel iteration step
// Solves Ax = b iteratively using Gauss-Seidel method (one iteration)
// input_A: coefficient matrix (N x N, row-major)
// input_b: right-hand side vector (N elements)
// input_x: current solution estimate (N elements)
// output_x: updated solution estimate (N elements)
void gauss_seidel_step_cpu(const float* __restrict__ input_A,
                            const float* __restrict__ input_b,
                            const float* __restrict__ input_x,
                            float* __restrict__ output_x,
                            const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        float sigma = 0.0f;
        
        // Sum using already updated values (from output_x)
        for (uint32_t j = 0; j < i; j++) {
            sigma += input_A[i * N + j] * output_x[j];
        }
        
        // Sum using old values (from input_x)
        for (uint32_t j = i + 1; j < N; j++) {
            sigma += input_A[i * N + j] * input_x[j];
        }
        
        // Update current element
        output_x[i] = (input_b[i] - sigma) / input_A[i * N + i];
    }
}

// Accelerator implementation of single Gauss-Seidel iteration step
LOOM_ACCEL()
void gauss_seidel_step_dsa(const float* __restrict__ input_A,
                            const float* __restrict__ input_b,
                            const float* __restrict__ input_x,
                            float* __restrict__ output_x,
                            const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t i = 0; i < N; i++) {
        float sigma = 0.0f;
        
        // Sum using already updated values
        for (uint32_t j = 0; j < i; j++) {
            sigma += input_A[i * N + j] * output_x[j];
        }
        
        // Sum using old values
        for (uint32_t j = i + 1; j < N; j++) {
            sigma += input_A[i * N + j] * input_x[j];
        }
        
        // Update current element
        output_x[i] = (input_b[i] - sigma) / input_A[i * N + i];
    }
}



