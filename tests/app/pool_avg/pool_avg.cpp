// Loom kernel implementation: pool_avg
#include "pool_avg.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: 2D average pooling
// Tests complete compilation chain with nested loops and 2D strided memory access
// Test: 4x4 input, 2x2 pool, stride 2 → [3.5, 5.5, 11.5, 13.5]






// CPU implementation of 2D average pooling
// Input: H x W image (row-major)
// Output: OH x OW image where OH = (H - pool_h) / stride_h + 1, OW = (W - pool_w) / stride_w + 1
void pool_avg_cpu(const float* __restrict__ input,
                  float* __restrict__ output,
                  const uint32_t H,
                  const uint32_t W,
                  const uint32_t pool_h,
                  const uint32_t pool_w,
                  const uint32_t stride_h,
                  const uint32_t stride_w) {
    uint32_t OH = (H - pool_h) / stride_h + 1;
    uint32_t OW = (W - pool_w) / stride_w + 1;
    float pool_size = static_cast<float>(pool_h * pool_w);
    
    for (uint32_t oh = 0; oh < OH; oh++) {
        for (uint32_t ow = 0; ow < OW; ow++) {
            float sum = 0.0f;
            
            for (uint32_t ph = 0; ph < pool_h; ph++) {
                for (uint32_t pw = 0; pw < pool_w; pw++) {
                    uint32_t h = oh * stride_h + ph;
                    uint32_t w = ow * stride_w + pw;
                    sum += input[h * W + w];
                }
            }
            
            output[oh * OW + ow] = sum / pool_size;
        }
    }
}

// Average pooling: output[oh,ow] = avg(input[oh*stride_h+ph, ow*stride_w+pw])
// Accelerator implementation of 2D average pooling
LOOM_ACCEL()
void pool_avg_dsa(LOOM_MEMORY_BANK(4, block) LOOM_STREAM const float* __restrict__ input,
                  LOOM_STREAM float* __restrict__ output,
                  const uint32_t H,
                  const uint32_t W,
                  const uint32_t pool_h,
                  const uint32_t pool_w,
                  const uint32_t stride_h,
                  const uint32_t stride_w) {
    uint32_t OH = (H - pool_h) / stride_h + 1;
    uint32_t OW = (W - pool_w) / stride_w + 1;
    float pool_size = static_cast<float>(pool_h * pool_w);
    
    for (uint32_t oh = 0; oh < OH; oh++) {
        for (uint32_t ow = 0; ow < OW; ow++) {
            float sum = 0.0f;
            
            for (uint32_t ph = 0; ph < pool_h; ph++) {
                for (uint32_t pw = 0; pw < pool_w; pw++) {
                    uint32_t h = oh * stride_h + ph;
                    uint32_t w = ow * stride_w + pw;
                    sum += input[h * W + w];
                }
            }
            
            output[oh * OW + ow] = sum / pool_size;
        }
    }
}





