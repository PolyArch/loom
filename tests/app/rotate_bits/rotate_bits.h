// Loom kernel: rotate_bits
#ifndef ROTATE_BITS_H
#define ROTATE_BITS_H

#include <cstdint>
#include <cstddef>

void rotate_bits_cpu(const uint32_t* __restrict__ input_data, const uint32_t* __restrict__ input_shift, uint32_t* __restrict__ output_result, const uint32_t N);

void rotate_bits_dsa(const uint32_t* __restrict__ input_data, const uint32_t* __restrict__ input_shift, uint32_t* __restrict__ output_result, const uint32_t N);

#endif // ROTATE_BITS_H
