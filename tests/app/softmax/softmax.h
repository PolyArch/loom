// Loom kernel: softmax
#ifndef SOFTMAX_H
#define SOFTMAX_H

#include <cstdint>
#include <cstddef>

void softmax_cpu(const float* __restrict__ input, float* __restrict__ output, const uint32_t N);

void softmax_dsa(const float* __restrict__ input, float* __restrict__ output, const uint32_t N);

#endif // SOFTMAX_H
