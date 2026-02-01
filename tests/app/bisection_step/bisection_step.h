// Loom kernel: bisection_step
#ifndef BISECTION_STEP_H
#define BISECTION_STEP_H

#include <cstdint>
#include <cstddef>

void bisection_step_cpu(const float* __restrict__ input_a, const float* __restrict__ input_b, const float* __restrict__ input_fa, const float* __restrict__ input_fb, const float* __restrict__ input_fc, float* __restrict__ output_a, float* __restrict__ output_b, const uint32_t N);

void bisection_step_dsa(const float* __restrict__ input_a, const float* __restrict__ input_b, const float* __restrict__ input_fa, const float* __restrict__ input_fb, const float* __restrict__ input_fc, float* __restrict__ output_a, float* __restrict__ output_b, const uint32_t N);

#endif // BISECTION_STEP_H
