// Loom kernel: variance
#ifndef VARIANCE_H
#define VARIANCE_H

#include <cstdint>
#include <cstddef>

float variance_cpu(const float* __restrict__ input, const uint32_t N);

float variance_dsa(const float* __restrict__ input, const uint32_t N);

#endif // VARIANCE_H
