// Loom kernel implementation: normalize_vec3
#include "normalize_vec3.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: 3D vector normalization
// Tests complete compilation chain with sqrt and floating-point division
// Test: 6 random 3D vectors → normalized to unit length







// CPU implementation of 3D vector normalization
// Normalizes N 3D vectors to unit length
// Each vector is 3 consecutive floats: (x, y, z)
// For zero vectors, outputs (0, 0, 0)
void normalize_vec3_cpu(const float* __restrict__ input_vec,
                        float* __restrict__ output_normalized,
                        const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        float x = input_vec[i * 3 + 0];
        float y = input_vec[i * 3 + 1];
        float z = input_vec[i * 3 + 2];
        
        float length = sqrtf(x * x + y * y + z * z);
        
        if (length > 1e-8f) {
            output_normalized[i * 3 + 0] = x / length;
            output_normalized[i * 3 + 1] = y / length;
            output_normalized[i * 3 + 2] = z / length;
        } else {
            output_normalized[i * 3 + 0] = 0.0f;
            output_normalized[i * 3 + 1] = 0.0f;
            output_normalized[i * 3 + 2] = 0.0f;
        }
    }
}

// Accelerator implementation of 3D vector normalization
LOOM_ACCEL()
void normalize_vec3_dsa(const float* __restrict__ input_vec,
                        float* __restrict__ output_normalized,
                        const uint32_t N) {
    LOOM_PARALLEL(4, contiguous)
    LOOM_TRIPCOUNT_FULL(256, 256, 1, 1024)
    for (uint32_t i = 0; i < N; i++) {
        float x = input_vec[i * 3 + 0];
        float y = input_vec[i * 3 + 1];
        float z = input_vec[i * 3 + 2];
        
        float length = sqrtf(x * x + y * y + z * z);
        
        if (length > 1e-8f) {
            output_normalized[i * 3 + 0] = x / length;
            output_normalized[i * 3 + 1] = y / length;
            output_normalized[i * 3 + 2] = z / length;
        } else {
            output_normalized[i * 3 + 0] = 0.0f;
            output_normalized[i * 3 + 1] = 0.0f;
            output_normalized[i * 3 + 2] = 0.0f;
        }
    }
}



