// Loom kernel: prefix_sum_exclusive
#ifndef PREFIX_SUM_EXCLUSIVE_H
#define PREFIX_SUM_EXCLUSIVE_H

#include <cstdint>
#include <cstddef>

void prefix_sum_exclusive_cpu(const uint32_t* __restrict__ input, uint32_t* __restrict__ output, const uint32_t N);

void prefix_sum_exclusive_dsa(const uint32_t* __restrict__ input, uint32_t* __restrict__ output, const uint32_t N);

#endif // PREFIX_SUM_EXCLUSIVE_H
