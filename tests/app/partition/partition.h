// Loom kernel: partition
#ifndef PARTITION_H
#define PARTITION_H

#include <cstdint>
#include <cstddef>

void partition_cpu(const float* __restrict__ input, float* __restrict__ output, uint32_t* __restrict__ output_pivot_idx, const uint32_t N, const float pivot);

void partition_dsa(const float* __restrict__ input, float* __restrict__ output, uint32_t* __restrict__ output_pivot_idx, const uint32_t N, const float pivot);

#endif // PARTITION_H
