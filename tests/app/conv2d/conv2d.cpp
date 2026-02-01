// Loom kernel implementation: conv2d
#include "conv2d.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: 2D convolution
// Tests complete compilation chain with 6-layer nested loops
// Test: input (1x4x4), kernel=[1,0,0,1] (1x1x2x2), stride=1 → output (1x3x3)






// CPU implementation of 2D convolution using direct method
// Input: C_in x H x W image (channel-major)
// Kernel: C_out x C_in x KH x KW weights (output channel, input channel, height, width)
// Output: C_out x OH x OW image where OH = (H - KH) / stride_h + 1, OW = (W - KW) / stride_w + 1
void conv2d_cpu(const float* __restrict__ input,
                const float* __restrict__ kernel,
                float* __restrict__ output,
                const uint32_t C_in,
                const uint32_t C_out,
                const uint32_t H,
                const uint32_t W,
                const uint32_t KH,
                const uint32_t KW,
                const uint32_t stride_h,
                const uint32_t stride_w) {
    uint32_t OH = (H - KH) / stride_h + 1;
    uint32_t OW = (W - KW) / stride_w + 1;
    
    // Initialize output to zero
    for (uint32_t i = 0; i < C_out * OH * OW; i++) {
        output[i] = 0.0f;
    }
    
    // Convolution
    for (uint32_t co = 0; co < C_out; co++) {
        for (uint32_t oh = 0; oh < OH; oh++) {
            for (uint32_t ow = 0; ow < OW; ow++) {
                float sum = 0.0f;
                
                for (uint32_t ci = 0; ci < C_in; ci++) {
                    for (uint32_t kh = 0; kh < KH; kh++) {
                        for (uint32_t kw = 0; kw < KW; kw++) {
                            uint32_t h = oh * stride_h + kh;
                            uint32_t w = ow * stride_w + kw;
                            
                            float input_val = input[ci * (H * W) + h * W + w];
                            float kernel_val = kernel[co * (C_in * KH * KW) + ci * (KH * KW) + kh * KW + kw];
                            sum += input_val * kernel_val;
                        }
                    }
                }
                
                output[co * (OH * OW) + oh * OW + ow] = sum;
            }
        }
    }
}

// Accelerator implementation of 2D convolution
LOOM_ACCEL()
void conv2d_dsa(LOOM_MEMORY_BANK(4, block) LOOM_STREAM const float* __restrict__ input,
                LOOM_STREAM const float* __restrict__ kernel,
                float* __restrict__ output,
                const uint32_t C_in,
                const uint32_t C_out,
                const uint32_t H,
                const uint32_t W,
                const uint32_t KH,
                const uint32_t KW,
                const uint32_t stride_h,
                const uint32_t stride_w) {
    uint32_t OH = (H - KH) / stride_h + 1;
    uint32_t OW = (W - KW) / stride_w + 1;
    
    // Initialize output to zero
    for (uint32_t i = 0; i < C_out * OH * OW; i++) {
        output[i] = 0.0f;
    }
    
    // Convolution
    LOOM_PARALLEL(4, contiguous)
    for (uint32_t co = 0; co < C_out; co++) {
        LOOM_UNROLL(4)
        for (uint32_t oh = 0; oh < OH; oh++) {
            LOOM_TRIPCOUNT_FULL(16, 16, 1, 64)
            for (uint32_t ow = 0; ow < OW; ow++) {
                float sum = 0.0f;
                
                for (uint32_t ci = 0; ci < C_in; ci++) {
                    for (uint32_t kh = 0; kh < KH; kh++) {
                        for (uint32_t kw = 0; kw < KW; kw++) {
                            uint32_t h = oh * stride_h + kh;
                            uint32_t w = ow * stride_w + kw;
                            
                            float input_val = input[ci * (H * W) + h * W + w];
                            float kernel_val = kernel[co * (C_in * KH * KW) + ci * (KH * KW) + kh * KW + kw];
                            sum += input_val * kernel_val;
                        }
                    }
                }
                
                output[co * (OH * OW) + oh * OW + ow] = sum;
            }
        }
    }
}


