// Loom kernel: sort_insertion
#ifndef SORT_INSERTION_H
#define SORT_INSERTION_H

#include <cstdint>
#include <cstddef>

void sort_insertion_cpu(const float* __restrict__ input, float* __restrict__ output, const uint32_t N);

void sort_insertion_dsa(const float* __restrict__ input, float* __restrict__ output, const uint32_t N);

#endif // SORT_INSERTION_H
