// Loom kernel: dot_product_3d
#ifndef DOT_PRODUCT_3D_H
#define DOT_PRODUCT_3D_H

#include <cstdint>
#include <cstddef>

void dot_product_3d_cpu(const float* __restrict__ input_vec_a, const float* __restrict__ input_vec_b, float* __restrict__ output_result, const uint32_t N);

void dot_product_3d_dsa(const float* __restrict__ input_vec_a, const float* __restrict__ input_vec_b, float* __restrict__ output_result, const uint32_t N);

#endif // DOT_PRODUCT_3D_H
