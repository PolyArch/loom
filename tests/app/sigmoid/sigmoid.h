// Loom kernel: sigmoid
#ifndef SIGMOID_H
#define SIGMOID_H

#include <cstdint>
#include <cstddef>

void sigmoid_cpu(const float* __restrict__ input, float* __restrict__ output, const uint32_t N);

void sigmoid_dsa(const float* __restrict__ input, float* __restrict__ output, const uint32_t N);

#endif // SIGMOID_H
