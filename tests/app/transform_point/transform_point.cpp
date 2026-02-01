// Loom kernel implementation: transform_point
#include "transform_point.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>







// CPU implementation of affine point transformation
// Applies 3D affine transformation: p' = M * p + t
// Transform matrix M is 3x3 (9 floats, row-major)
// Translation vector t is 3 floats
// Transforms N 3D points
void transform_point_cpu(const float* __restrict__ input_points,
                         const float* __restrict__ input_matrix,
                         const float* __restrict__ input_translation,
                         float* __restrict__ output_points,
                         const uint32_t N) {
    // Extract matrix elements
    float m00 = input_matrix[0], m01 = input_matrix[1], m02 = input_matrix[2];
    float m10 = input_matrix[3], m11 = input_matrix[4], m12 = input_matrix[5];
    float m20 = input_matrix[6], m21 = input_matrix[7], m22 = input_matrix[8];
    
    // Extract translation
    float tx = input_translation[0];
    float ty = input_translation[1];
    float tz = input_translation[2];
    
    for (uint32_t i = 0; i < N; i++) {
        float px = input_points[i * 3 + 0];
        float py = input_points[i * 3 + 1];
        float pz = input_points[i * 3 + 2];
        
        output_points[i * 3 + 0] = m00*px + m01*py + m02*pz + tx;
        output_points[i * 3 + 1] = m10*px + m11*py + m12*pz + ty;
        output_points[i * 3 + 2] = m20*px + m21*py + m22*pz + tz;
    }
}

// Accelerator implementation of affine point transformation
LOOM_ACCEL()
void transform_point_dsa(const float* __restrict__ input_points,
                         const float* __restrict__ input_matrix,
                         const float* __restrict__ input_translation,
                         float* __restrict__ output_points,
                         const uint32_t N) {
    // Extract matrix elements
    float m00 = input_matrix[0], m01 = input_matrix[1], m02 = input_matrix[2];
    float m10 = input_matrix[3], m11 = input_matrix[4], m12 = input_matrix[5];
    float m20 = input_matrix[6], m21 = input_matrix[7], m22 = input_matrix[8];
    
    // Extract translation
    float tx = input_translation[0];
    float ty = input_translation[1];
    float tz = input_translation[2];
    
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t i = 0; i < N; i++) {
        float px = input_points[i * 3 + 0];
        float py = input_points[i * 3 + 1];
        float pz = input_points[i * 3 + 2];
        
        output_points[i * 3 + 0] = m00*px + m01*py + m02*pz + tx;
        output_points[i * 3 + 1] = m10*px + m11*py + m12*pz + ty;
        output_points[i * 3 + 2] = m20*px + m21*py + m22*pz + tz;
    }
}



