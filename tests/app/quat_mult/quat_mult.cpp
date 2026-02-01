// Loom kernel implementation: quat_mult
#include "quat_mult.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Quaternion multiplication
// Tests complete compilation chain with quaternion arithmetic

// Test: Multiple quaternion pairs (N=3) with various rotation values

// CPU implementation of quaternion multiplication
// Multiplies N pairs of quaternions: q = q1 * q2
// Each quaternion is stored as (w, x, y, z) - 4 consecutive floats
// Formula: q = q1 * q2
//   w = w1*w2 - x1*x2 - y1*y2 - z1*z2
//   x = w1*x2 + x1*w2 + y1*z2 - z1*y2
//   y = w1*y2 - x1*z2 + y1*w2 + z1*x2
//   z = w1*z2 + x1*y2 - y1*x2 + z1*w2
void quat_mult_cpu(const float* __restrict__ input_quat_1,
                   const float* __restrict__ input_quat_2,
                   float* __restrict__ output_quat,
                   const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        float w1 = input_quat_1[i * 4 + 0];
        float x1 = input_quat_1[i * 4 + 1];
        float y1 = input_quat_1[i * 4 + 2];
        float z1 = input_quat_1[i * 4 + 3];

        float w2 = input_quat_2[i * 4 + 0];
        float x2 = input_quat_2[i * 4 + 1];
        float y2 = input_quat_2[i * 4 + 2];
        float z2 = input_quat_2[i * 4 + 3];

        output_quat[i * 4 + 0] = w1*w2 - x1*x2 - y1*y2 - z1*z2;
        output_quat[i * 4 + 1] = w1*x2 + x1*w2 + y1*z2 - z1*y2;
        output_quat[i * 4 + 2] = w1*y2 - x1*z2 + y1*w2 + z1*x2;
        output_quat[i * 4 + 3] = w1*z2 + x1*y2 - y1*x2 + z1*w2;
    }
}

// Accelerator implementation of quaternion multiplication
LOOM_ACCEL()
void quat_mult_dsa(const float* __restrict__ input_quat_1,
                   const float* __restrict__ input_quat_2,
                   float* __restrict__ output_quat,
                   const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t i = 0; i < N; i++) {
        float w1 = input_quat_1[i * 4 + 0];
        float x1 = input_quat_1[i * 4 + 1];
        float y1 = input_quat_1[i * 4 + 2];
        float z1 = input_quat_1[i * 4 + 3];

        float w2 = input_quat_2[i * 4 + 0];
        float x2 = input_quat_2[i * 4 + 1];
        float y2 = input_quat_2[i * 4 + 2];
        float z2 = input_quat_2[i * 4 + 3];

        output_quat[i * 4 + 0] = w1*w2 - x1*x2 - y1*y2 - z1*z2;
        output_quat[i * 4 + 1] = w1*x2 + x1*w2 + y1*z2 - z1*y2;
        output_quat[i * 4 + 2] = w1*y2 - x1*z2 + y1*w2 + z1*x2;
        output_quat[i * 4 + 3] = w1*z2 + x1*y2 - y1*x2 + z1*w2;
    }
}

