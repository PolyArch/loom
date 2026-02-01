// Loom kernel: spmspm
#ifndef SPMSPM_H
#define SPMSPM_H

#include <cstdint>

void spmspm_cpu(const uint32_t* __restrict__ A_values,
                const uint32_t* __restrict__ A_col_indices,
                const uint32_t* __restrict__ A_row_ptr,
                const uint32_t* __restrict__ B_values,
                const uint32_t* __restrict__ B_col_indices,
                const uint32_t* __restrict__ B_row_ptr,
                uint32_t* __restrict__ C_values,
                uint32_t* __restrict__ C_col_indices,
                uint32_t* __restrict__ C_row_ptr,
                uint32_t* __restrict__ temp_row,
                const uint32_t M,
                const uint32_t N,
                const uint32_t K);

void spmspm_dsa(const uint32_t* __restrict__ A_values,
                const uint32_t* __restrict__ A_col_indices,
                const uint32_t* __restrict__ A_row_ptr,
                const uint32_t* __restrict__ B_values,
                const uint32_t* __restrict__ B_col_indices,
                const uint32_t* __restrict__ B_row_ptr,
                uint32_t* __restrict__ C_values,
                uint32_t* __restrict__ C_col_indices,
                uint32_t* __restrict__ C_row_ptr,
                uint32_t* __restrict__ temp_row,
                const uint32_t M,
                const uint32_t N,
                const uint32_t K);

#endif // SPMSPM_H
