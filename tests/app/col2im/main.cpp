#include <cstdio>

#include "col2im.h"
#include <cmath>

int main() {
    const uint32_t C = 3;
    const uint32_t H = 8;
    const uint32_t W = 8;
    const uint32_t KH = 3;
    const uint32_t KW = 3;
    const uint32_t stride_h = 1;
    const uint32_t stride_w = 1;
    const uint32_t OH = (H - KH) / stride_h + 1;
    const uint32_t OW = (W - KW) / stride_w + 1;
    const uint32_t input_rows = C * KH * KW;
    const uint32_t input_cols = OH * OW;
    
    // Input column matrix
    float input[input_rows * input_cols];
    
    // Output images
    float expect_output[C * H * W];
    float calculated_output[C * H * W];
    
    // Initialize input column matrix
    for (uint32_t i = 0; i < input_rows * input_cols; i++) {
        input[i] = static_cast<float>(i % 10);
    }
    
    // Compute expected result with CPU version
    col2im_cpu(input, expect_output, C, H, W, KH, KW, stride_h, stride_w);
    
    // Compute result with accelerator version
    col2im_dsa(input, calculated_output, C, H, W, KH, KW, stride_h, stride_w);
    
    // Compare results
    for (uint32_t i = 0; i < C * H * W; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > 1e-5f) {
            printf("col2im: FAILED\n");
            return 1;
        }
    }
    
    printf("col2im: PASSED\n");
    return 0;
}

