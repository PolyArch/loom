// Loom kernel implementation: cross_product
#include "cross_product.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: 3D cross product
// Tests complete compilation chain with strided memory access and arithmetic operations
// Test: a=[(1,0,0),(0,1,0)], b=[(0,1,0),(0,0,1)], N=2 → result=[(0,0,1),(1,0,0)]






// CPU implementation of 3D cross product
// Computes cross product for N pairs of 3D vectors
// Each vector is 3 consecutive floats: (x, y, z)
// Result: a × b = (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx)
void cross_product_cpu(const float* __restrict__ input_vec_a,
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
        
        output_result[i * 3 + 0] = ay * bz - az * by;
        output_result[i * 3 + 1] = az * bx - ax * bz;
        output_result[i * 3 + 2] = ax * by - ay * bx;
    }
}

// Accelerator implementation of 3D cross product
LOOM_ACCEL()
void cross_product_dsa(const float* __restrict__ input_vec_a,
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
        
        output_result[i * 3 + 0] = ay * bz - az * by;
        output_result[i * 3 + 1] = az * bx - ax * bz;
        output_result[i * 3 + 2] = ax * by - ay * bx;
    }
}


// Read x, y, z components separately using strided 2D reads (stride=3)
// Cross product arithmetic: a × b = (ay*bz - az*by, az*bx - ax*bz, ax*by - ay*bx)
// Write results back using strided 2D writes (stride=3)



