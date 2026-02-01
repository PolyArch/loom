// Loom kernel: sbox_lookup
#ifndef SBOX_LOOKUP_H
#define SBOX_LOOKUP_H

#include <cstdint>
#include <cstddef>

void sbox_lookup_cpu(const uint32_t* __restrict__ input_data, const uint32_t* __restrict__ input_sbox, uint32_t* __restrict__ output_result, const uint32_t N);

void sbox_lookup_dsa(const uint32_t* __restrict__ input_data, const uint32_t* __restrict__ input_sbox, uint32_t* __restrict__ output_result, const uint32_t N);

#endif // SBOX_LOOKUP_H
