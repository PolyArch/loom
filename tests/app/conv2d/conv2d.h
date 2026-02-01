// Loom kernel: conv2d
#ifndef CONV2D_H
#define CONV2D_H

#include <cstdint>
#include <cstddef>

void conv2d_cpu(const float* __restrict__ input, const float* __restrict__ kernel, float* __restrict__ output, const uint32_t C_in, const uint32_t C_out, const uint32_t H, const uint32_t W, const uint32_t KH, const uint32_t KW, const uint32_t stride_h, const uint32_t stride_w);

void conv2d_dsa(const float* __restrict__ input, const float* __restrict__ kernel, float* __restrict__ output, const uint32_t C_in, const uint32_t C_out, const uint32_t H, const uint32_t W, const uint32_t KH, const uint32_t KW, const uint32_t stride_h, const uint32_t stride_w);

#endif // CONV2D_H
