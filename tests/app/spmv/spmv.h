// Loom kernel: spmv
#ifndef SPMV_H
#define SPMV_H

#include <cstdint>
#include <cstddef>

void spmv_cpu(const uint32_t* __restrict__ values, const uint32_t* __restrict__ col_indices, const uint32_t* __restrict__ row_ptr, const uint32_t* __restrict__ x, uint32_t* __restrict__ y, const uint32_t M, const uint32_t N);

void spmv_dsa(const uint32_t* __restrict__ values, const uint32_t* __restrict__ col_indices, const uint32_t* __restrict__ row_ptr, const uint32_t* __restrict__ x, uint32_t* __restrict__ y, const uint32_t M, const uint32_t N);

#endif // SPMV_H
