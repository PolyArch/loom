// Loom kernel: compare_swap
#ifndef COMPARE_SWAP_H
#define COMPARE_SWAP_H

#include <cstdint>
#include <cstddef>

void compare_swap_cpu(const float* __restrict__ input_a, const float* __restrict__ input_b, float* __restrict__ output_min, float* __restrict__ output_max, const uint32_t N);

void compare_swap_dsa(const float* __restrict__ input_a, const float* __restrict__ input_b, float* __restrict__ output_min, float* __restrict__ output_max, const uint32_t N);

#endif // COMPARE_SWAP_H
