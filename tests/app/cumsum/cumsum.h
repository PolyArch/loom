// Loom kernel: cumsum
#ifndef CUMSUM_H
#define CUMSUM_H

#include <cstdint>
#include <cstddef>

void cumsum_cpu(const float* __restrict__ input, float* __restrict__ output, const uint32_t N);

void cumsum_dsa(const float* __restrict__ input, float* __restrict__ output, const uint32_t N);

#endif // CUMSUM_H
