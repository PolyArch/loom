// Loom kernel: lower_bound
#ifndef LOWER_BOUND_H
#define LOWER_BOUND_H

#include <cstdint>
#include <cstddef>

void lower_bound_cpu(const float* __restrict__ input_sorted, const float* __restrict__ input_targets, uint32_t* __restrict__ output_indices, const uint32_t N, const uint32_t M);

void lower_bound_dsa(const float* __restrict__ input_sorted, const float* __restrict__ input_targets, uint32_t* __restrict__ output_indices, const uint32_t N, const uint32_t M);

#endif // LOWER_BOUND_H
