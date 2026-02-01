#include <cstdio>

#include "transform_point.h"
#include <cmath>

int main() {
    const uint32_t N = 64;
    
    // Input points
    float input_points[N * 3];
    for (uint32_t i = 0; i < N; i++) {
        input_points[i * 3 + 0] = 1.0f + i * 0.1f;
        input_points[i * 3 + 1] = 2.0f + i * 0.2f;
        input_points[i * 3 + 2] = 3.0f + i * 0.3f;
    }
    
    // Transformation matrix (identity with scaling)
    float matrix[9] = {
        2.0f, 0.0f, 0.0f,
        0.0f, 2.0f, 0.0f,
        0.0f, 0.0f, 2.0f
    };
    
    // Translation vector
    float translation[3] = {1.0f, 2.0f, 3.0f};
    
    // Output arrays
    float expect_result[N * 3];
    float calculated_result[N * 3];
    
    // Compute expected result with CPU version
    transform_point_cpu(input_points, matrix, translation, expect_result, N);
    
    // Compute result with accelerator version
    transform_point_dsa(input_points, matrix, translation, calculated_result, N);
    
    // Compare results with tolerance
    for (uint32_t i = 0; i < N * 3; i++) {
        if (fabsf(expect_result[i] - calculated_result[i]) > 1e-5f) {
            printf("transform_point: FAILED\n");
            return 1;
        }
    }
    
    printf("transform_point: PASSED\n");
    return 0;
}

