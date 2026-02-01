// Loom kernel: upper_bound
#ifndef UPPER_BOUND_H
#define UPPER_BOUND_H

#include <cstdint>
#include <cstddef>

void upper_bound_cpu(const float* __restrict__ input_sorted, const float* __restrict__ input_targets, uint32_t* __restrict__ output_indices, const uint32_t N, const uint32_t M);

void upper_bound_dsa(const float* __restrict__ input_sorted, const float* __restrict__ input_targets, uint32_t* __restrict__ output_indices, const uint32_t N, const uint32_t M);

#endif // UPPER_BOUND_H
