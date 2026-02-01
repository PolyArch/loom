// Loom kernel: gf_mul
#ifndef GF_MUL_H
#define GF_MUL_H

#include <cstdint>
#include <cstddef>

void gf_mul_cpu(const uint32_t* __restrict__ input_A, const uint32_t* __restrict__ input_B, uint32_t* __restrict__ output_C, const uint32_t N);

void gf_mul_dsa(const uint32_t* __restrict__ input_A, const uint32_t* __restrict__ input_B, uint32_t* __restrict__ output_C, const uint32_t N);

#endif // GF_MUL_H
