// Loom kernel: runge_kutta_step
#ifndef RUNGE_KUTTA_STEP_H
#define RUNGE_KUTTA_STEP_H

#include <cstdint>
#include <cstddef>

void runge_kutta_step_cpu(const float* __restrict__ input_y, const float* __restrict__ input_k1, const float* __restrict__ input_k2, const float* __restrict__ input_k3, const float* __restrict__ input_k4, float* __restrict__ output_y, const float h, const uint32_t N);

void runge_kutta_step_dsa(const float* __restrict__ input_y, const float* __restrict__ input_k1, const float* __restrict__ input_k2, const float* __restrict__ input_k3, const float* __restrict__ input_k4, float* __restrict__ output_y, const float h, const uint32_t N);

#endif // RUNGE_KUTTA_STEP_H
