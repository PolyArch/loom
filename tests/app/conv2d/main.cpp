#include <cstdio>

#include "conv2d.h"
#include <cmath>

int main() {
    const uint32_t C_in = 3;
    const uint32_t C_out = 4;
    const uint32_t H = 8;
    const uint32_t W = 8;
    const uint32_t KH = 3;
    const uint32_t KW = 3;
    const uint32_t stride_h = 1;
    const uint32_t stride_w = 1;
    const uint32_t OH = (H - KH) / stride_h + 1;
    const uint32_t OW = (W - KW) / stride_w + 1;
    
    // Input image
    float input[C_in * H * W];
    
    // Convolution kernel
    float kernel[C_out * C_in * KH * KW];
    
    // Output feature maps
    float expect_output[C_out * OH * OW];
    float calculated_output[C_out * OH * OW];
    
    // Initialize input image
    for (uint32_t i = 0; i < C_in * H * W; i++) {
        input[i] = static_cast<float>(i % 10);
    }
    
    // Initialize kernel weights
    for (uint32_t i = 0; i < C_out * C_in * KH * KW; i++) {
        kernel[i] = (static_cast<float>(i % 5) - 2.0f) / 10.0f;
    }
    
    // Compute expected result with CPU version
    conv2d_cpu(input, kernel, expect_output, C_in, C_out, H, W, KH, KW, stride_h, stride_w);
    
    // Compute result with accelerator version
    conv2d_dsa(input, kernel, calculated_output, C_in, C_out, H, W, KH, KW, stride_h, stride_w);
    
    // Compare results
    for (uint32_t i = 0; i < C_out * OH * OW; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > 1e-4f) {
            printf("conv2d: FAILED\n");
            return 1;
        }
    }
    
    printf("conv2d: PASSED\n");
    return 0;
}

