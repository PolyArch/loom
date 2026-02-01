// Loom kernel: pool_avg
#ifndef POOL_AVG_H
#define POOL_AVG_H

#include <cstdint>
#include <cstddef>

void pool_avg_cpu(const float* __restrict__ input, float* __restrict__ output, const uint32_t H, const uint32_t W, const uint32_t pool_h, const uint32_t pool_w, const uint32_t stride_h, const uint32_t stride_w);

void pool_avg_dsa(const float* __restrict__ input, float* __restrict__ output, const uint32_t H, const uint32_t W, const uint32_t pool_h, const uint32_t pool_w, const uint32_t stride_h, const uint32_t stride_w);

#endif // POOL_AVG_H
