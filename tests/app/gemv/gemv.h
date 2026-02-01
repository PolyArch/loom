// Loom kernel: gemv
#ifndef GEMV_H
#define GEMV_H

#include <cstdint>
#include <cstddef>

void gemv_cpu(const uint32_t alpha, const uint32_t* __restrict__ A, const uint32_t* __restrict__ x, const uint32_t beta, const uint32_t* __restrict__ input_y, uint32_t* __restrict__ output_y, const uint32_t M, const uint32_t N);

void gemv_dsa(const uint32_t alpha, const uint32_t* __restrict__ A, const uint32_t* __restrict__ x, const uint32_t beta, const uint32_t* __restrict__ input_y, uint32_t* __restrict__ output_y, const uint32_t M, const uint32_t N);

#endif // GEMV_H
