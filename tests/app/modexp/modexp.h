// Loom kernel: modexp
#ifndef MODEXP_H
#define MODEXP_H

#include <cstdint>
#include <cstddef>

void modexp_cpu(const uint32_t* __restrict__ input_base, const uint32_t* __restrict__ input_exp, uint32_t* __restrict__ output_result, const uint32_t modulus, const uint32_t N);

void modexp_dsa(const uint32_t* __restrict__ input_base, const uint32_t* __restrict__ input_exp, uint32_t* __restrict__ output_result, const uint32_t modulus, const uint32_t N);

#endif // MODEXP_H
