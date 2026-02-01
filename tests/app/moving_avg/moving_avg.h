// Loom kernel: moving_avg
#ifndef MOVING_AVG_H
#define MOVING_AVG_H

#include <cstdint>
#include <cstddef>

void moving_avg_cpu(const float* __restrict__ input, float* __restrict__ output, const uint32_t N, const uint32_t window_size);

void moving_avg_dsa(const float* __restrict__ input, float* __restrict__ output, const uint32_t N, const uint32_t window_size);

#endif // MOVING_AVG_H
