// Loom kernel: unpack_bits
#ifndef UNPACK_BITS_H
#define UNPACK_BITS_H

#include <cstdint>
#include <cstddef>

void unpack_bits_cpu(const uint32_t* __restrict__ input_packed, uint32_t* __restrict__ output_bits, const uint32_t num_bits);

void unpack_bits_dsa(const uint32_t* __restrict__ input_packed, uint32_t* __restrict__ output_bits, const uint32_t num_bits);

#endif // UNPACK_BITS_H
