// Loom kernel: sort_quick
#ifndef SORT_QUICK_H
#define SORT_QUICK_H

#include <cstdint>
#include <cstddef>

void sort_quick_cpu(const float* __restrict__ input, float* __restrict__ output, const uint32_t N);

void sort_quick_dsa(const float* __restrict__ input, float* __restrict__ output, const uint32_t N);

#endif // SORT_QUICK_H
