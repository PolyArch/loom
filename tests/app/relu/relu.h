// Loom kernel: relu
#ifndef RELU_H
#define RELU_H

#include <cstdint>
#include <cstddef>

void relu_cpu(const float* __restrict__ input, float* __restrict__ output, const uint32_t N);

void relu_dsa(const float* __restrict__ input, float* __restrict__ output, const uint32_t N);

#endif // RELU_H
