// Loom kernel: cross_product
#ifndef CROSS_PRODUCT_H
#define CROSS_PRODUCT_H

#include <cstdint>
#include <cstddef>

void cross_product_cpu(const float* __restrict__ input_vec_a, const float* __restrict__ input_vec_b, float* __restrict__ output_result, const uint32_t N);

void cross_product_dsa(const float* __restrict__ input_vec_a, const float* __restrict__ input_vec_b, float* __restrict__ output_result, const uint32_t N);

#endif // CROSS_PRODUCT_H
