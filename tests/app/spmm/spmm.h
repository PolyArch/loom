// Loom kernel: spmm
#ifndef SPMM_H
#define SPMM_H

#include <cstdint>
#include <cstddef>

void spmm_cpu(const uint32_t* __restrict__ A_values, const uint32_t* __restrict__ A_col_indices, const uint32_t* __restrict__ A_row_ptr, const uint32_t* __restrict__ B, uint32_t* __restrict__ C, const uint32_t M, const uint32_t N, const uint32_t K);

void spmm_dsa(const uint32_t* __restrict__ A_values, const uint32_t* __restrict__ A_col_indices, const uint32_t* __restrict__ A_row_ptr, const uint32_t* __restrict__ B, uint32_t* __restrict__ C, const uint32_t M, const uint32_t N, const uint32_t K);

#endif // SPMM_H
