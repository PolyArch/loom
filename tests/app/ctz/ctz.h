// Loom kernel: ctz
#ifndef CTZ_H
#define CTZ_H

#include <cstdint>
#include <cstddef>

void ctz_cpu(const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_count, const uint32_t N);

void ctz_dsa(const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_count, const uint32_t N);

#endif // CTZ_H
