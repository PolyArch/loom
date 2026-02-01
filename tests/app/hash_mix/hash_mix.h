// Loom kernel: hash_mix
#ifndef HASH_MIX_H
#define HASH_MIX_H

#include <cstdint>
#include <cstddef>

void hash_mix_cpu(const uint32_t* __restrict__ input_state, const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_state, const uint32_t N);

void hash_mix_dsa(const uint32_t* __restrict__ input_state, const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_state, const uint32_t N);

#endif // HASH_MIX_H
