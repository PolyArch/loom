// Loom kernel: normalize_vec3
#ifndef NORMALIZE_VEC3_H
#define NORMALIZE_VEC3_H

#include <cstdint>
#include <cstddef>

void normalize_vec3_cpu(const float* __restrict__ input_vec, float* __restrict__ output_normalized, const uint32_t N);

void normalize_vec3_dsa(const float* __restrict__ input_vec, float* __restrict__ output_normalized, const uint32_t N);

#endif // NORMALIZE_VEC3_H
