#include <cstdio>

#include "batchnorm.h"
#include <cmath>

int main() {
    const uint32_t C = 4;
    const uint32_t H = 8;
    const uint32_t W = 8;
    const float epsilon = 1e-5f;
    
    // Input image
    float input[C * H * W];
    
    // Batch norm parameters (per channel)
    float mean[C];
    float variance[C];
    float gamma[C];
    float beta[C];
    
    // Output images
    float expect_output[C * H * W];
    float calculated_output[C * H * W];
    
    // Initialize input image
    for (uint32_t i = 0; i < C * H * W; i++) {
        input[i] = static_cast<float>(i % 100) - 50.0f;
    }
    
    // Initialize batch norm parameters
    for (uint32_t c = 0; c < C; c++) {
        mean[c] = static_cast<float>(c * 10);
        variance[c] = static_cast<float>(c + 1) * 2.0f;
        gamma[c] = 1.0f;
        beta[c] = 0.0f;
    }
    
    // Compute expected result with CPU version
    batchnorm_cpu(input, mean, variance, gamma, beta, expect_output, C, H, W, epsilon);
    
    // Compute result with accelerator version
    batchnorm_dsa(input, mean, variance, gamma, beta, calculated_output, C, H, W, epsilon);
    
    // Compare results
    for (uint32_t i = 0; i < C * H * W; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > 1e-4f) {
            printf("batchnorm: FAILED\n");
            return 1;
        }
    }
    
    printf("batchnorm: PASSED\n");
    return 0;
}

