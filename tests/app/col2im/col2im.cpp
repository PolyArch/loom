// Loom kernel implementation: col2im
#include "col2im.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Col2im (column to image) transformation
// Tests complete compilation chain with 5-layer nested loops and accumulation
// Test: input=[1..36] (4x9), C=1, H=4, W=4, KH=2, KW=2, stride=1 → output (1x4x4)

// CPU implementation of col2im (column to image) transformation
// Inverse of im2col, accumulates overlapping patches back to image
// Input: (C * KH * KW) x (OH * OW) column matrix
// Output: C x H x W image (channel-major)
void col2im_cpu(const float* __restrict__ input,
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

    // Initialize output to zero (accumulation)
    for (uint32_t i = 0; i < C * H * W; i++) {
        output[i] = 0.0f;
    }

    for (uint32_t c = 0; c < C; c++) {
        for (uint32_t kh = 0; kh < KH; kh++) {
            for (uint32_t kw = 0; kw < KW; kw++) {
                uint32_t row = c * (KH * KW) + kh * KW + kw;

                for (uint32_t oh = 0; oh < OH; oh++) {
                    for (uint32_t ow = 0; ow < OW; ow++) {
                        uint32_t h = oh * stride_h + kh;
                        uint32_t w = ow * stride_w + kw;
                        uint32_t col = oh * OW + ow;

                        output[c * (H * W) + h * W + w] += input[row * (OH * OW) + col];
                    }
                }
            }
        }
    }
}

// Accelerator implementation of col2im
LOOM_ACCEL()
void col2im_dsa(LOOM_MEMORY_BANK(4, block) LOOM_STREAM const float* __restrict__ input,
                LOOM_STREAM float* __restrict__ output,
                const uint32_t C,
                const uint32_t H,
                const uint32_t W,
                const uint32_t KH,
                const uint32_t KW,
                const uint32_t stride_h,
                const uint32_t stride_w) {
    uint32_t OH = (H - KH) / stride_h + 1;
    uint32_t OW = (W - KW) / stride_w + 1;

    // Initialize output to zero (accumulation)
    for (uint32_t i = 0; i < C * H * W; i++) {
        output[i] = 0.0f;
    }

    LOOM_PARALLEL(4, contiguous)
    for (uint32_t c = 0; c < C; c++) {
        LOOM_UNROLL(4)
        for (uint32_t kh = 0; kh < KH; kh++) {
            LOOM_TRIPCOUNT_FULL(16, 16, 1, 64)
            for (uint32_t kw = 0; kw < KW; kw++) {
                uint32_t row = c * (KH * KW) + kh * KW + kw;

                for (uint32_t oh = 0; oh < OH; oh++) {
                    for (uint32_t ow = 0; ow < OW; ow++) {
                        uint32_t h = oh * stride_h + kh;
                        uint32_t w = ow * stride_w + kw;
                        uint32_t col = oh * OW + ow;

                        output[c * (H * W) + h * W + w] += input[row * (OH * OW) + col];
                    }
                }
            }
        }
    }
}

