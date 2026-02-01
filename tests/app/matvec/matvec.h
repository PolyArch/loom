// Loom kernel: matvec
#ifndef MATVEC_H
#define MATVEC_H

#include <cstdint>
#include <cstddef>

void matvec_cpu(const uint32_t* __restrict__ A, const uint32_t* __restrict__ x, uint32_t* __restrict__ y, const uint32_t M, const uint32_t N);

void matvec_dsa(const uint32_t* __restrict__ A, const uint32_t* __restrict__ x, uint32_t* __restrict__ y, const uint32_t M, const uint32_t N);

#endif // MATVEC_H
