// Loom kernel: upsample
#ifndef UPSAMPLE_H
#define UPSAMPLE_H

#include <cstdint>
#include <cstddef>

void upsample_cpu(const float* __restrict__ input, float* __restrict__ output, const uint32_t input_size, const uint32_t factor);

void upsample_dsa(const float* __restrict__ input, float* __restrict__ output, const uint32_t input_size, const uint32_t factor);

#endif // UPSAMPLE_H
