// Loom kernel: depthwise_conv
#ifndef DEPTHWISE_CONV_H
#define DEPTHWISE_CONV_H

#include <cstdint>
#include <cstddef>

void depthwise_conv_cpu(const float* __restrict__ input, const float* __restrict__ kernel, float* __restrict__ output, const uint32_t C, const uint32_t H, const uint32_t W, const uint32_t KH, const uint32_t KW, const uint32_t stride_h, const uint32_t stride_w);

void depthwise_conv_dsa(const float* __restrict__ input, const float* __restrict__ kernel, float* __restrict__ output, const uint32_t C, const uint32_t H, const uint32_t W, const uint32_t KH, const uint32_t KW, const uint32_t stride_h, const uint32_t stride_w);

#endif // DEPTHWISE_CONV_H
