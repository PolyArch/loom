// Loom kernel: matmul
#ifndef MATMUL_H
#define MATMUL_H

#include <cstdint>
#include <cstddef>

void matmul_cpu(const uint32_t* __restrict__ A, const uint32_t* __restrict__ B, uint32_t* __restrict__ C, const uint32_t M, const uint32_t N, const uint32_t K);

void matmul_dsa(const uint32_t* __restrict__ A, const uint32_t* __restrict__ B, uint32_t* __restrict__ C, const uint32_t M, const uint32_t N, const uint32_t K);

#endif // MATMUL_H
