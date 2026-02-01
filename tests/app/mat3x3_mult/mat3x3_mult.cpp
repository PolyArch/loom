// Loom kernel implementation: mat3x3_mult
#include "mat3x3_mult.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: 3x3 matrix multiplication
// Tests complete compilation chain with fixed-size matrix operations
// Test: 3 pairs of random 3x3 matrices → C = A * B






// CPU implementation of 3x3 matrix multiplication
// Multiplies N pairs of 3x3 matrices: C = A * B
// Each matrix is stored in row-major order (9 consecutive floats)
void mat3x3_mult_cpu(const float* __restrict__ input_mat_a,
                     const float* __restrict__ input_mat_b,
                     float* __restrict__ output_mat_c,
                     const uint32_t N) {
    for (uint32_t n = 0; n < N; n++) {
        const float* A = &input_mat_a[n * 9];
        const float* B = &input_mat_b[n * 9];
        float* C = &output_mat_c[n * 9];
        
        // Compute C = A * B
        for (uint32_t i = 0; i < 3; i++) {
            for (uint32_t j = 0; j < 3; j++) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < 3; k++) {
                    sum += A[i * 3 + k] * B[k * 3 + j];
                }
                C[i * 3 + j] = sum;
            }
        }
    }
}

// Accelerator implementation of 3x3 matrix multiplication
LOOM_ACCEL()
void mat3x3_mult_dsa(const float* __restrict__ input_mat_a,
                     const float* __restrict__ input_mat_b,
                     float* __restrict__ output_mat_c,
                     const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t n = 0; n < N; n++) {
        const float* A = &input_mat_a[n * 9];
        const float* B = &input_mat_b[n * 9];
        float* C = &output_mat_c[n * 9];
        
        // Compute C = A * B
        for (uint32_t i = 0; i < 3; i++) {
            for (uint32_t j = 0; j < 3; j++) {
                float sum = 0.0f;
                for (uint32_t k = 0; k < 3; k++) {
                    sum += A[i * 3 + k] * B[k * 3 + j];
                }
                C[i * 3 + j] = sum;
            }
        }
    }
}



