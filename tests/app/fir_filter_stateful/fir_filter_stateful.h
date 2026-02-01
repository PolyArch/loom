// Loom kernel: fir_filter_stateful
#ifndef FIR_FILTER_STATEFUL_H
#define FIR_FILTER_STATEFUL_H

#include <cstdint>

void fir_filter_stateful_cpu(const float* __restrict__ input,
                              const float* __restrict__ coeffs,
                              const float* __restrict__ input_state,
                              float* __restrict__ output,
                              float* __restrict__ output_state,
                              float* __restrict__ current_state,
                              const uint32_t input_size,
                              const uint32_t num_taps);

void fir_filter_stateful_dsa(const float* __restrict__ input,
                              const float* __restrict__ coeffs,
                              const float* __restrict__ input_state,
                              float* __restrict__ output,
                              float* __restrict__ output_state,
                              float* __restrict__ current_state,
                              const uint32_t input_size,
                              const uint32_t num_taps);

#endif // FIR_FILTER_STATEFUL_H
