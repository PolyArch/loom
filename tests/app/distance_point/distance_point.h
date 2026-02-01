// Loom kernel: distance_point
#ifndef DISTANCE_POINT_H
#define DISTANCE_POINT_H

#include <cstdint>
#include <cstddef>

void distance_point_cpu(const float* __restrict__ input_point_a, const float* __restrict__ input_point_b, float* __restrict__ output_distance, const uint32_t N);

void distance_point_dsa(const float* __restrict__ input_point_a, const float* __restrict__ input_point_b, float* __restrict__ output_distance, const uint32_t N);

#endif // DISTANCE_POINT_H
