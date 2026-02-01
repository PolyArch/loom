// Loom kernel: integrate_trapz
#ifndef INTEGRATE_TRAPZ_H
#define INTEGRATE_TRAPZ_H

#include <cstdint>
#include <cstddef>

float integrate_trapz_cpu(const float* __restrict__ input_x, const float* __restrict__ input_y, const uint32_t N);

float integrate_trapz_dsa(const float* __restrict__ input_x, const float* __restrict__ input_y, const uint32_t N);

#endif // INTEGRATE_TRAPZ_H
