// Loom kernel: sort_bubble
#ifndef SORT_BUBBLE_H
#define SORT_BUBBLE_H

#include <cstdint>
#include <cstddef>

void sort_bubble_cpu(const float* __restrict__ input, float* __restrict__ output, const uint32_t N);

void sort_bubble_dsa(const float* __restrict__ input, float* __restrict__ output, const uint32_t N);

#endif // SORT_BUBBLE_H
