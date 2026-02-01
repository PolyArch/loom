// Loom kernel: binary_search
#ifndef BINARY_SEARCH_H
#define BINARY_SEARCH_H

#include <cstdint>
#include <cstddef>

void binary_search_cpu(const float* __restrict__ input_sorted, const float* __restrict__ input_targets, uint32_t* __restrict__ output_indices, const uint32_t N, const uint32_t M);

void binary_search_dsa(const float* __restrict__ input_sorted, const float* __restrict__ input_targets, uint32_t* __restrict__ output_indices, const uint32_t N, const uint32_t M);

#endif // BINARY_SEARCH_H
