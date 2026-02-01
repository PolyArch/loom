// Loom kernel: interpolate_linear
#ifndef INTERPOLATE_LINEAR_H
#define INTERPOLATE_LINEAR_H

#include <cstdint>
#include <cstddef>

void interpolate_linear_cpu(const float* __restrict__ input_x, const float* __restrict__ input_y, const float* __restrict__ input_xq, float* __restrict__ output_yq, const uint32_t N_data, const uint32_t N_query);

void interpolate_linear_dsa(const float* __restrict__ input_x, const float* __restrict__ input_y, const float* __restrict__ input_xq, float* __restrict__ output_yq, const uint32_t N_data, const uint32_t N_query);

#endif // INTERPOLATE_LINEAR_H
