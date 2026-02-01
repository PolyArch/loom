// Loom kernel: byte_swap
#ifndef BYTE_SWAP_H
#define BYTE_SWAP_H

#include <cstdint>
#include <cstddef>

void byte_swap_cpu(const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_swapped, const uint32_t N);

void byte_swap_dsa(const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_swapped, const uint32_t N);

#endif // BYTE_SWAP_H
