// Loom kernel: popcount
#ifndef POPCOUNT_H
#define POPCOUNT_H

#include <cstdint>
#include <cstddef>

void popcount_cpu(const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_count, const uint32_t N);

void popcount_dsa(const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_count, const uint32_t N);

#endif // POPCOUNT_H
