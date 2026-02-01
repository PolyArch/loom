// Loom kernel: transform_point
#ifndef TRANSFORM_POINT_H
#define TRANSFORM_POINT_H

#include <cstdint>
#include <cstddef>

void transform_point_cpu(const float* __restrict__ input_points, const float* __restrict__ input_matrix, const float* __restrict__ input_translation, float* __restrict__ output_points, const uint32_t N);

void transform_point_dsa(const float* __restrict__ input_points, const float* __restrict__ input_matrix, const float* __restrict__ input_translation, float* __restrict__ output_points, const uint32_t N);

#endif // TRANSFORM_POINT_H
