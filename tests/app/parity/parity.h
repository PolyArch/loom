// Loom kernel: parity
#ifndef PARITY_H
#define PARITY_H

#include <cstdint>
#include <cstddef>

void parity_cpu(const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_parity, const uint32_t N);

void parity_dsa(const uint32_t* __restrict__ input_data, uint32_t* __restrict__ output_parity, const uint32_t N);

#endif // PARITY_H
