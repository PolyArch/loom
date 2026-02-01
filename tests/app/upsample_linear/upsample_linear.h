// Loom kernel: upsample_linear
#ifndef UPSAMPLE_LINEAR_H
#define UPSAMPLE_LINEAR_H

#include <cstdint>
#include <cstddef>

void upsample_linear_cpu(const float* __restrict__ input, float* __restrict__ output, const uint32_t input_size, const uint32_t factor);

void upsample_linear_dsa(const float* __restrict__ input, float* __restrict__ output, const uint32_t input_size, const uint32_t factor);

#endif // UPSAMPLE_LINEAR_H
