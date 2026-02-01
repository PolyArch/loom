#include <cstdio>

#include "normalize_vec3.h"
#include <cmath>

int main() {
    const uint32_t N = 64;
    
    // Input vectors (N 3D vectors)
    float input[N * 3];
    
    for (uint32_t i = 0; i < N; i++) {
        input[i * 3 + 0] = 3.0f + i * 0.1f;
        input[i * 3 + 1] = 4.0f + i * 0.2f;
        input[i * 3 + 2] = 0.0f;
    }
    
    // Special case: zero vector
    input[0] = 0.0f;
    input[1] = 0.0f;
    input[2] = 0.0f;
    
    // Output arrays
    float expect_normalized[N * 3];
    float calculated_normalized[N * 3];
    
    // Compute expected result with CPU version
    normalize_vec3_cpu(input, expect_normalized, N);
    
    // Compute result with accelerator version
    normalize_vec3_dsa(input, calculated_normalized, N);
    
    // Compare results with tolerance
    for (uint32_t i = 0; i < N * 3; i++) {
        if (fabsf(expect_normalized[i] - calculated_normalized[i]) > 1e-5f) {
            printf("normalize_vec3: FAILED\n");
            return 1;
        }
    }
    
    printf("normalize_vec3: PASSED\n");
    return 0;
}

