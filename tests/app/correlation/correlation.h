// Loom kernel: correlation
#ifndef CORRELATION_H
#define CORRELATION_H

#include <cstdint>
#include <cstddef>

void correlation_cpu(const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ output, const uint32_t x_size, const uint32_t y_size);

void correlation_dsa(const float* __restrict__ x, const float* __restrict__ y, float* __restrict__ output, const uint32_t x_size, const uint32_t y_size);

#endif // CORRELATION_H
