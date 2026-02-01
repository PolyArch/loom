// Loom kernel: bit_reverse
#ifndef BIT_REVERSE_H
#define BIT_REVERSE_H

#include <cstdint>
#include <cstddef>

void bit_reverse_cpu(const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_reversed, const uint32_t N);

void bit_reverse_dsa(const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_reversed, const uint32_t N);

#endif // BIT_REVERSE_H
