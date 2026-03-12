// Loom kernel: sort_merge
#ifndef SORT_MERGE_H
#define SORT_MERGE_H

#include <cstdint>
#include <cstddef>

void sort_merge_cpu(const float* __restrict__ input, float* __restrict__ output, float* __restrict__ temp, const uint32_t N);

void sort_merge_dsa(const float* __restrict__ input, float* __restrict__ output, float* __restrict__ temp, const uint32_t N);

#endif // SORT_MERGE_H
