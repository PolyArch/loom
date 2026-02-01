// Loom kernel: prefix_sum_inclusive
#ifndef PREFIX_SUM_INCLUSIVE_H
#define PREFIX_SUM_INCLUSIVE_H

#include <cstdint>
#include <cstddef>

void prefix_sum_inclusive_cpu(const uint32_t* __restrict__ input, uint32_t* __restrict__ output, const uint32_t N);

void prefix_sum_inclusive_dsa(const uint32_t* __restrict__ input, uint32_t* __restrict__ output, const uint32_t N);

#endif // PREFIX_SUM_INCLUSIVE_H
