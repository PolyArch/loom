// Loom kernel: modmul
#ifndef MODMUL_H
#define MODMUL_H

#include <cstdint>
#include <cstddef>

void modmul_cpu(const uint32_t* __restrict__ input_A, const uint32_t* __restrict__ input_B, uint32_t* __restrict__ output_C, const uint32_t modulus, const uint32_t N);

void modmul_dsa(const uint32_t* __restrict__ input_A, const uint32_t* __restrict__ input_B, uint32_t* __restrict__ output_C, const uint32_t modulus, const uint32_t N);

#endif // MODMUL_H
