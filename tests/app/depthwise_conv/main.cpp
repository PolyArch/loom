#include <cstdio>

#include "depthwise_conv.h"
#include <cmath>

int main() {
    const uint32_t C = 4;
    const uint32_t H = 8;
    const uint32_t W = 8;
    const uint32_t KH = 3;
    const uint32_t KW = 3;
    const uint32_t stride_h = 1;
    const uint32_t stride_w = 1;
    const uint32_t OH = (H - KH) / stride_h + 1;
    const uint32_t OW = (W - KW) / stride_w + 1;
    
    // Input image
    float input[C * H * W];
    
    // Depthwise kernel (one per channel)
    float kernel[C * KH * KW];
    
    // Output feature maps
    float expect_output[C * OH * OW];
    float calculated_output[C * OH * OW];
    
    // Initialize input image
    for (uint32_t i = 0; i < C * H * W; i++) {
        input[i] = static_cast<float>(i % 10);
    }
    
    // Initialize kernel weights
    for (uint32_t i = 0; i < C * KH * KW; i++) {
        kernel[i] = (static_cast<float>(i % 5) - 2.0f) / 10.0f;
    }
    
    // Compute expected result with CPU version
    depthwise_conv_cpu(input, kernel, expect_output, C, H, W, KH, KW, stride_h, stride_w);
    
    // Compute result with accelerator version
    depthwise_conv_dsa(input, kernel, calculated_output, C, H, W, KH, KW, stride_h, stride_w);
    
    // Compare results
    for (uint32_t i = 0; i < C * OH * OW; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > 1e-4f) {
            printf("depthwise_conv: FAILED\n");
            return 1;
        }
    }
    
    printf("depthwise_conv: PASSED\n");
    return 0;
}

