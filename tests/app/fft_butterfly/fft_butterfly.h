// Loom kernel: fft_butterfly
#ifndef FFT_BUTTERFLY_H
#define FFT_BUTTERFLY_H

#include <cstdint>
#include <cstddef>

void fft_butterfly_cpu(const float* __restrict__ input_real, const float* __restrict__ input_imag, float* __restrict__ output_real, float* __restrict__ output_imag, const uint32_t N);

void fft_butterfly_dsa(const float* __restrict__ input_real, const float* __restrict__ input_imag, float* __restrict__ output_real, float* __restrict__ output_imag, const uint32_t N);

#endif // FFT_BUTTERFLY_H
