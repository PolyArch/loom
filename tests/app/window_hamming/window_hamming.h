// Loom kernel: window_hamming
#ifndef WINDOW_HAMMING_H
#define WINDOW_HAMMING_H

#include <cstdint>
#include <cstddef>

void window_hamming_cpu(const float* __restrict__ input, float* __restrict__ output, const uint32_t N);

void window_hamming_dsa(const float* __restrict__ input, float* __restrict__ output, const uint32_t N);

#endif // WINDOW_HAMMING_H
