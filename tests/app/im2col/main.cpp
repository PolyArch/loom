#include <cstdio>

#include "im2col.h"
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
    const uint32_t output_rows = C * KH * KW;
    const uint32_t output_cols = OH * OW;
    
    // Input image
    float input[C * H * W];
    
    // Output column matrices
    float expect_output[output_rows * output_cols];
    float calculated_output[output_rows * output_cols];
    
    // Initialize input image
    for (uint32_t i = 0; i < C * H * W; i++) {
        input[i] = static_cast<float>(i);
    }
    
    // Compute expected result with CPU version
    im2col_cpu(input, expect_output, C, H, W, KH, KW, stride_h, stride_w);
    
    // Compute result with accelerator version
    im2col_dsa(input, calculated_output, C, H, W, KH, KW, stride_h, stride_w);
    
    // Compare results
    for (uint32_t i = 0; i < output_rows * output_cols; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > 1e-6f) {
            printf("im2col: FAILED\n");
            return 1;
        }
    }
    
    printf("im2col: PASSED\n");
    return 0;
}

