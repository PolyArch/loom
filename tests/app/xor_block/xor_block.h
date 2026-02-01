// Loom kernel: xor_block
#ifndef XOR_BLOCK_H
#define XOR_BLOCK_H

#include <cstdint>
#include <cstddef>

void xor_block_cpu(const uint32_t* __restrict__ input_A, const uint32_t* __restrict__ input_B, uint32_t* __restrict__ output_C, const uint32_t N);

void xor_block_dsa(const uint32_t* __restrict__ input_A, const uint32_t* __restrict__ input_B, uint32_t* __restrict__ output_C, const uint32_t N);

#endif // XOR_BLOCK_H
