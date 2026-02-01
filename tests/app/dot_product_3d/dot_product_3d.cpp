// Loom kernel implementation: dot_product_3d
#include "dot_product_3d.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: 3D dot product
// Tests complete compilation chain with strided memory access for 3D vectors
// Test: a=[(1,2,3),(4,5,6)], b=[(1,0,0),(0,1,0)], N=2 → result=[1.0,5.0]

// CPU implementation of 3D dot product
// Computes dot product for N pairs of 3D vectors
// Each vector is 3 consecutive floats: (x, y, z)
void dot_product_3d_cpu(const float* __restrict__ input_vec_a,
                        const float* __restrict__ input_vec_b,
                        float* __restrict__ output_result,
                        const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        float ax = input_vec_a[i * 3 + 0];
        float ay = input_vec_a[i * 3 + 1];
        float az = input_vec_a[i * 3 + 2];

        float bx = input_vec_b[i * 3 + 0];
        float by = input_vec_b[i * 3 + 1];
        float bz = input_vec_b[i * 3 + 2];

        output_result[i] = ax * bx + ay * by + az * bz;
    }
}

// Accelerator implementation of 3D dot product
LOOM_ACCEL()
void dot_product_3d_dsa(const float* __restrict__ input_vec_a,
                        const float* __restrict__ input_vec_b,
                        float* __restrict__ output_result,
                        const uint32_t N) {
    LOOM_PARALLEL(4, contiguous)
    LOOM_TRIPCOUNT_FULL(256, 256, 1, 1024)
    for (uint32_t i = 0; i < N; i++) {
        float ax = input_vec_a[i * 3 + 0];
        float ay = input_vec_a[i * 3 + 1];
        float az = input_vec_a[i * 3 + 2];

        float bx = input_vec_b[i * 3 + 0];
        float by = input_vec_b[i * 3 + 1];
        float bz = input_vec_b[i * 3 + 2];

        output_result[i] = ax * bx + ay * by + az * bz;
    }
}

// Read x, y, z components separately using strided 2D reads (stride=3)
// Dot product arithmetic: a · b = ax*bx + ay*by + az*bz
// Write results back

