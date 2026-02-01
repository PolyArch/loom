// Loom kernel: window_hanning
#ifndef WINDOW_HANNING_H
#define WINDOW_HANNING_H

#include <cstdint>
#include <cstddef>

void window_hanning_cpu(const float* __restrict__ input, float* __restrict__ output, const uint32_t N);

void window_hanning_dsa(const float* __restrict__ input, float* __restrict__ output, const uint32_t N);

#endif // WINDOW_HANNING_H
