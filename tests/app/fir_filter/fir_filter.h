// FIR Filter kernel header
// FIR: output[n] = sum(coeffs[k] * input[n-k]) for k = 0 to num_taps-1
#ifndef FIR_FILTER_H
#define FIR_FILTER_H

#include <cstdint>

// CPU reference implementation
void fir_filter_cpu(const float *__restrict input, const float *__restrict coeffs,
                    float *__restrict output, uint32_t input_size,
                    uint32_t num_taps);

// DSA accelerated implementation with Loom pragmas
void fir_filter_dsa(const float *__restrict input,
                    const float *__restrict coeffs, float *__restrict output,
                    uint32_t input_size, uint32_t num_taps);

#endif // FIR_FILTER_H
