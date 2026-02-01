// Loom kernel: transpose
#ifndef TRANSPOSE_H
#define TRANSPOSE_H

#include <cstdint>
#include <cstddef>

void transpose_cpu(const uint32_t* __restrict__ A, uint32_t* __restrict__ B, const uint32_t M, const uint32_t N);

void transpose_dsa(const uint32_t* __restrict__ A, uint32_t* __restrict__ B, const uint32_t M, const uint32_t N);

#endif // TRANSPOSE_H
