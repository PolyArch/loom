// Loom kernel: clz
#ifndef CLZ_H
#define CLZ_H

#include <cstdint>
#include <cstddef>

void clz_cpu(const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_count, const uint32_t N);

void clz_dsa(const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_count, const uint32_t N);

#endif // CLZ_H
