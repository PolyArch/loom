// Loom kernel: quantile
#ifndef QUANTILE_H
#define QUANTILE_H

#include <cstdint>
#include <cstddef>

float quantile_cpu(const float* __restrict__ sorted_input, const uint32_t N, const float q);

float quantile_dsa(const float* __restrict__ sorted_input, const uint32_t N, const float q);

#endif // QUANTILE_H
