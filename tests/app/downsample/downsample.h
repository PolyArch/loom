// Loom kernel: downsample
#ifndef DOWNSAMPLE_H
#define DOWNSAMPLE_H

#include <cstdint>
#include <cstddef>

void downsample_cpu(const float* __restrict__ input, float* __restrict__ output, const uint32_t input_size, const uint32_t factor);

void downsample_dsa(const float* __restrict__ input, float* __restrict__ output, const uint32_t input_size, const uint32_t factor);

#endif // DOWNSAMPLE_H
