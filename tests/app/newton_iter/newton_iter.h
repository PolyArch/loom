// Loom kernel: newton_iter
#ifndef NEWTON_ITER_H
#define NEWTON_ITER_H

#include <cstdint>
#include <cstddef>

void newton_iter_cpu(const float* __restrict__ input_x, const float* __restrict__ input_f, const float* __restrict__ input_df, float* __restrict__ output_x, const uint32_t N);

void newton_iter_dsa(const float* __restrict__ input_x, const float* __restrict__ input_f, const float* __restrict__ input_df, float* __restrict__ output_x, const uint32_t N);

#endif // NEWTON_ITER_H
