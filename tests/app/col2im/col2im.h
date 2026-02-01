// Loom kernel: col2im
#ifndef COL2IM_H
#define COL2IM_H

#include <cstdint>
#include <cstddef>

void col2im_cpu(const float* __restrict__ input, float* __restrict__ output, const uint32_t C, const uint32_t H, const uint32_t W, const uint32_t KH, const uint32_t KW, const uint32_t stride_h, const uint32_t stride_w);

void col2im_dsa(const float* __restrict__ input, float* __restrict__ output, const uint32_t C, const uint32_t H, const uint32_t W, const uint32_t KH, const uint32_t KW, const uint32_t stride_h, const uint32_t stride_w);

#endif // COL2IM_H
