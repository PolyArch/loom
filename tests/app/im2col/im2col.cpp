// Loom kernel implementation: im2col
#include "im2col.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Image to column (im2col) transformation
// Tests complete compilation chain with complex indexing for convolution
// Test: 1x3x3 image, 2x2 kernel, stride=1 → 4x4 column matrix

// CPU implementation of im2col (image to column) transformation
// Transforms input image into column matrix for convolution via matrix multiplication
// Input: C x H x W image (channel-major: all C channels, then H rows, then W cols)
// Output: (C * KH * KW) x (OH * OW) matrix where OH, OW are output dimensions
// KH, KW: kernel height and width
// stride_h, stride_w: convolution stride
void im2col_cpu(const float* __restrict__ input,
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
        for (uint32_t kh = 0; kh < KH; kh++) {
            for (uint32_t kw = 0; kw < KW; kw++) {
                uint32_t row = c * (KH * KW) + kh * KW + kw;

                for (uint32_t oh = 0; oh < OH; oh++) {
                    for (uint32_t ow = 0; ow < OW; ow++) {
                        uint32_t h = oh * stride_h + kh;
                        uint32_t w = ow * stride_w + kw;
                        uint32_t col = oh * OW + ow;

                        output[row * (OH * OW) + col] = input[c * (H * W) + h * W + w];
                    }
                }
            }
        }
    }
}

// Accelerator implementation of im2col
LOOM_ACCEL()
void im2col_dsa(LOOM_MEMORY_BANK(4, block) LOOM_STREAM const float* __restrict__ input,
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

                        output[row * (OH * OW) + col] = input[c * (H * W) + h * W + w];
                    }
                }
            }
        }
    }
}

