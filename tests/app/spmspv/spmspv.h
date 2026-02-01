// Loom kernel: spmspv
#ifndef SPMSPV_H
#define SPMSPV_H

#include <cstdint>

void spmspv_cpu(const uint32_t* __restrict__ A_values,
                const uint32_t* __restrict__ A_col_indices,
                const uint32_t* __restrict__ A_row_ptr,
                const uint32_t* __restrict__ x_values,
                const uint32_t* __restrict__ x_indices,
                const uint32_t x_nnz,
                uint32_t* __restrict__ y,
                uint32_t* __restrict__ x_dense,
                const uint32_t M,
                const uint32_t N);

void spmspv_dsa(const uint32_t* __restrict__ A_values,
                const uint32_t* __restrict__ A_col_indices,
                const uint32_t* __restrict__ A_row_ptr,
                const uint32_t* __restrict__ x_values,
                const uint32_t* __restrict__ x_indices,
                const uint32_t x_nnz,
                uint32_t* __restrict__ y,
                uint32_t* __restrict__ x_dense,
                const uint32_t M,
                const uint32_t N);

#endif // SPMSPV_H
