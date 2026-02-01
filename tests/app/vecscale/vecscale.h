// Loom kernel: vecscale
#ifndef VECSCALE_H
#define VECSCALE_H

#include <cstdint>
#include <cstddef>

void vecscale_cpu(const uint32_t* __restrict__ A, const uint32_t alpha, uint32_t* __restrict__ B, const uint32_t N);

void vecscale_dsa(const uint32_t* __restrict__ A, const uint32_t alpha, uint32_t* __restrict__ B, const uint32_t N);

#endif // VECSCALE_H
