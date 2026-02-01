// Loom kernel: mean
#ifndef MEAN_H
#define MEAN_H

#include <cstdint>
#include <cstddef>

float mean_cpu(const float* __restrict__ input, const uint32_t N);

float mean_dsa(const float* __restrict__ input, const uint32_t N);

#endif // MEAN_H
