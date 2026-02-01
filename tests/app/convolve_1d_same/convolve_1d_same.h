// Loom kernel: convolve_1d_same
#ifndef CONVOLVE_1D_SAME_H
#define CONVOLVE_1D_SAME_H

#include <cstdint>
#include <cstddef>

void convolve_1d_same_cpu(const float* __restrict__ input, const float* __restrict__ kernel, float* __restrict__ output, const uint32_t input_size, const uint32_t kernel_size);

void convolve_1d_same_dsa(const float* __restrict__ input, const float* __restrict__ kernel, float* __restrict__ output, const uint32_t input_size, const uint32_t kernel_size);

#endif // CONVOLVE_1D_SAME_H
