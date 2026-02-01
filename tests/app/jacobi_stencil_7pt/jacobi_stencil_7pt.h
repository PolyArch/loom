// Loom kernel: jacobi_stencil_7pt
#ifndef JACOBI_STENCIL_7PT_H
#define JACOBI_STENCIL_7PT_H

#include <cstdint>
#include <cstddef>

void jacobi_stencil_7pt_cpu(const float* __restrict__ input_grid, float* __restrict__ output_grid, const uint32_t L, const uint32_t M, const uint32_t N);

void jacobi_stencil_7pt_dsa(const float* __restrict__ input_grid, float* __restrict__ output_grid, const uint32_t L, const uint32_t M, const uint32_t N);

#endif // JACOBI_STENCIL_7PT_H
