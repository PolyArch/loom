// Loom kernel: ifft_butterfly
#ifndef IFFT_BUTTERFLY_H
#define IFFT_BUTTERFLY_H

#include <cstdint>
#include <cstddef>

void ifft_butterfly_cpu(const float* __restrict__ input_real, const float* __restrict__ input_imag, float* __restrict__ output_real, float* __restrict__ output_imag, const uint32_t N);

void ifft_butterfly_dsa(const float* __restrict__ input_real, const float* __restrict__ input_imag, float* __restrict__ output_real, float* __restrict__ output_imag, const uint32_t N);

#endif // IFFT_BUTTERFLY_H
