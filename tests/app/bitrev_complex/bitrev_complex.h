// Loom kernel: bitrev_complex
#ifndef BITREV_COMPLEX_H
#define BITREV_COMPLEX_H

#include <cstdint>
#include <cstddef>

void bitrev_complex_cpu(const float* __restrict__ input_real, const float* __restrict__ input_imag, float* __restrict__ output_real, float* __restrict__ output_imag, const uint32_t N);

void bitrev_complex_dsa(const float* __restrict__ input_real, const float* __restrict__ input_imag, float* __restrict__ output_real, float* __restrict__ output_imag, const uint32_t N);

#endif // BITREV_COMPLEX_H
