// Loom kernel implementation: depthwise_conv
#include "depthwise_conv.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Depthwise separable convolution
// Tests complete compilation chain with per-channel convolution
// Test: input (2x3x3), kernel (2x2x2), stride=1 → output (2x2x2)=[6,8,12,14,48,52,60,64]






// CPU implementation of depthwise separable convolution
// Each input channel is convolved with its own kernel (no mixing across channels)
// Input: C x H x W image (channel-major)
// Kernel: C x KH x KW weights (one kernel per channel)
// Output: C x OH x OW image where OH = (H - KH) / stride_h + 1, OW = (W - KW) / stride_w + 1
void depthwise_conv_cpu(const float* __restrict__ input,
                        const float* __restrict__ kernel,
                        float* __restrict__ output,
                        const uint32_t C,
                        const uint32_t H,
                        const uint32_t W,
                        const uint32_t KH,
                        const uint32_t KW,
                        const uint32_t stride_h,
                        const uint32_t stride_w) {
    uint32_t OH = (H - KH) / stride_h + 1;
    uint32_t OW = (W - KW) / stride_w + 1;
    
    for (uint32_t c = 0; c < C; c++) {
        for (uint32_t oh = 0; oh < OH; oh++) {
            for (uint32_t ow = 0; ow < OW; ow++) {
                float sum = 0.0f;
                
                for (uint32_t kh = 0; kh < KH; kh++) {
                    for (uint32_t kw = 0; kw < KW; kw++) {
                        uint32_t h = oh * stride_h + kh;
                        uint32_t w = ow * stride_w + kw;
                        
                        float input_val = input[c * (H * W) + h * W + w];
                        float kernel_val = kernel[c * (KH * KW) + kh * KW + kw];
                        sum += input_val * kernel_val;
                    }
                }
                
                output[c * (OH * OW) + oh * OW + ow] = sum;
            }
        }
    }
}

// Accelerator implementation of depthwise separable convolution
LOOM_ACCEL()
void depthwise_conv_dsa(LOOM_MEMORY_BANK(4, block) LOOM_STREAM const float* __restrict__ input,
                        LOOM_STREAM const float* __restrict__ kernel,
                        float* __restrict__ output,
                        const uint32_t C,
                        const uint32_t H,
                        const uint32_t W,
                        const uint32_t KH,
                        const uint32_t KW,
                        const uint32_t stride_h,
                        const uint32_t stride_w) {
    uint32_t OH = (H - KH) / stride_h + 1;
    uint32_t OW = (W - KW) / stride_w + 1;
    
    LOOM_PARALLEL(4, contiguous)
    for (uint32_t c = 0; c < C; c++) {
        LOOM_UNROLL(4)
        for (uint32_t oh = 0; oh < OH; oh++) {
            LOOM_TRIPCOUNT_FULL(16, 16, 1, 64)
            for (uint32_t ow = 0; ow < OW; ow++) {
                float sum = 0.0f;
                
                for (uint32_t kh = 0; kh < KH; kh++) {
                    for (uint32_t kw = 0; kw < KW; kw++) {
                        uint32_t h = oh * stride_h + kh;
                        uint32_t w = ow * stride_w + kw;
                        
                        float input_val = input[c * (H * W) + h * W + w];
                        float kernel_val = kernel[c * (KH * KW) + kh * KW + kw];
                        sum += input_val * kernel_val;
                    }
                }
                
                output[c * (OH * OW) + oh * OW + ow] = sum;
            }
        }
    }
}



