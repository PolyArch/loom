// Loom kernel: mat3x3_mult
#ifndef MAT3X3_MULT_H
#define MAT3X3_MULT_H

#include <cstdint>
#include <cstddef>

void mat3x3_mult_cpu(const float* __restrict__ input_mat_a, const float* __restrict__ input_mat_b, float* __restrict__ output_mat_c, const uint32_t N);

void mat3x3_mult_dsa(const float* __restrict__ input_mat_a, const float* __restrict__ input_mat_b, float* __restrict__ output_mat_c, const uint32_t N);

#endif // MAT3X3_MULT_H
