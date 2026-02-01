// Loom kernel: window_blackman
#ifndef WINDOW_BLACKMAN_H
#define WINDOW_BLACKMAN_H

#include <cstdint>
#include <cstddef>

void window_blackman_cpu(const float* __restrict__ input, float* __restrict__ output, const uint32_t N);

void window_blackman_dsa(const float* __restrict__ input, float* __restrict__ output, const uint32_t N);

#endif // WINDOW_BLACKMAN_H
