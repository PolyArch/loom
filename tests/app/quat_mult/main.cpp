#include <cstdio>

#include "quat_mult.h"
#include <cmath>

int main() {
    const uint32_t N = 64;
    
    // Input quaternions (N pairs of quaternions)
    float quat_1[N * 4];
    float quat_2[N * 4];
    
    for (uint32_t i = 0; i < N; i++) {
        // Quaternion 1: identity with small variation
        quat_1[i * 4 + 0] = 1.0f;           // w
        quat_1[i * 4 + 1] = 0.1f * i;       // x
        quat_1[i * 4 + 2] = 0.0f;           // y
        quat_1[i * 4 + 3] = 0.0f;           // z
        
        // Quaternion 2: rotation around z-axis
        quat_2[i * 4 + 0] = 0.9f;           // w
        quat_2[i * 4 + 1] = 0.0f;           // x
        quat_2[i * 4 + 2] = 0.0f;           // y
        quat_2[i * 4 + 3] = 0.1f * i;       // z
    }
    
    // Output arrays
    float expect_result[N * 4];
    float calculated_result[N * 4];
    
    // Compute expected result with CPU version
    quat_mult_cpu(quat_1, quat_2, expect_result, N);
    
    // Compute result with accelerator version
    quat_mult_dsa(quat_1, quat_2, calculated_result, N);
    
    // Compare results with tolerance
    for (uint32_t i = 0; i < N * 4; i++) {
        if (fabsf(expect_result[i] - calculated_result[i]) > 1e-5f) {
            printf("quat_mult: FAILED\n");
            return 1;
        }
    }
    
    printf("quat_mult: PASSED\n");
    return 0;
}

