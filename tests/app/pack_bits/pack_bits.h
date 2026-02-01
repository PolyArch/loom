// Loom kernel: pack_bits
#ifndef PACK_BITS_H
#define PACK_BITS_H

#include <cstdint>
#include <cstddef>

void pack_bits_cpu(const uint32_t* __restrict__ input_bits, uint32_t* __restrict__ output_packed, const uint32_t num_bits);

void pack_bits_dsa(const uint32_t* __restrict__ input_bits, uint32_t* __restrict__ output_packed, const uint32_t num_bits);

#endif // PACK_BITS_H
