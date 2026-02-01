// Loom kernel: im2col
#ifndef IM2COL_H
#define IM2COL_H

#include <cstdint>
#include <cstddef>

void im2col_cpu(const float* __restrict__ input, float* __restrict__ output, const uint32_t C, const uint32_t H, const uint32_t W, const uint32_t KH, const uint32_t KW, const uint32_t stride_h, const uint32_t stride_w);

void im2col_dsa(const float* __restrict__ input, float* __restrict__ output, const uint32_t C, const uint32_t H, const uint32_t W, const uint32_t KH, const uint32_t KW, const uint32_t stride_h, const uint32_t stride_w);

#endif // IM2COL_H
