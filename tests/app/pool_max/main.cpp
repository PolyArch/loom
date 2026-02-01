#include <cstdio>

#include "pool_max.h"
#include <cmath>

int main() {
    const uint32_t H = 32;
    const uint32_t W = 32;
    const uint32_t pool_h = 2;
    const uint32_t pool_w = 2;
    const uint32_t stride_h = 2;
    const uint32_t stride_w = 2;
    const uint32_t OH = (H - pool_h) / stride_h + 1;
    const uint32_t OW = (W - pool_w) / stride_w + 1;
    
    // Input image
    float input[H * W];
    
    // Output images
    float expect_output[OH * OW];
    float calculated_output[OH * OW];
    
    // Initialize input image
    for (uint32_t i = 0; i < H * W; i++) {
        input[i] = static_cast<float>(i % 100);
    }
    
    // Compute expected result with CPU version
    pool_max_cpu(input, expect_output, H, W, pool_h, pool_w, stride_h, stride_w);
    
    // Compute result with accelerator version
    pool_max_dsa(input, calculated_output, H, W, pool_h, pool_w, stride_h, stride_w);
    
    // Compare results
    for (uint32_t i = 0; i < OH * OW; i++) {
        if (fabsf(expect_output[i] - calculated_output[i]) > 1e-6f) {
            printf("pool_max: FAILED\n");
            return 1;
        }
    }
    
    printf("pool_max: PASSED\n");
    return 0;
}

