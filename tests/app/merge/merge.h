// Loom kernel: merge
#ifndef MERGE_H
#define MERGE_H

#include <cstdint>
#include <cstddef>

void merge_cpu(const float* __restrict__ input_a, const float* __restrict__ input_b, float* __restrict__ output, const uint32_t N, const uint32_t M);

void merge_dsa(const float* __restrict__ input_a, const float* __restrict__ input_b, float* __restrict__ output, const uint32_t N, const uint32_t M);

#endif // MERGE_H
