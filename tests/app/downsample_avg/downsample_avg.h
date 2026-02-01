// Loom kernel: downsample_avg
#ifndef DOWNSAMPLE_AVG_H
#define DOWNSAMPLE_AVG_H

#include <cstdint>
#include <cstddef>

void downsample_avg_cpu(const float* __restrict__ input, float* __restrict__ output, const uint32_t input_size, const uint32_t factor);

void downsample_avg_dsa(const float* __restrict__ input, float* __restrict__ output, const uint32_t input_size, const uint32_t factor);

#endif // DOWNSAMPLE_AVG_H
