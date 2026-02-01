// Loom kernel: gauss_seidel_step
#ifndef GAUSS_SEIDEL_STEP_H
#define GAUSS_SEIDEL_STEP_H

#include <cstdint>
#include <cstddef>

void gauss_seidel_step_cpu(const float* __restrict__ input_A, const float* __restrict__ input_b, const float* __restrict__ input_x, float* __restrict__ output_x, const uint32_t N);

void gauss_seidel_step_dsa(const float* __restrict__ input_A, const float* __restrict__ input_b, const float* __restrict__ input_x, float* __restrict__ output_x, const uint32_t N);

#endif // GAUSS_SEIDEL_STEP_H
