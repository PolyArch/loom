// FIR Filter kernel implementations
// FIR: output[n] = sum(coeffs[k] * input[n-k]) for k = 0 to num_taps-1

#include "fir_filter.h"
#include "loom/loom.h"

// CPU reference implementation
void fir_filter_cpu(const float *__restrict input, const float *__restrict coeffs,
                    float *__restrict output, uint32_t input_size,
                    uint32_t num_taps) {
  for (uint32_t n = 0; n < input_size; n++) {
    float sum = 0.0f;
    for (uint32_t k = 0; k < num_taps; k++) {
      int32_t idx = static_cast<int32_t>(n) - static_cast<int32_t>(k);
      if (idx >= 0) {
        sum += coeffs[k] * input[idx];
      }
    }
    output[n] = sum;
  }
}

// DSA accelerated implementation with Loom pragmas
LOOM_TARGET("temporal")
LOOM_ACCEL("fir_filter")
void fir_filter_dsa(const float *__restrict input,
                    const float *__restrict coeffs, float *__restrict output,
                    uint32_t input_size, uint32_t num_taps) {
  outer_loop:
  LOOM_PARALLEL(4)
  for (uint32_t n = 0; n < input_size; n++) {
    LOOM_REDUCE(+)
    float sum = 0.0f;

    inner_loop:
    for (uint32_t k = 0; k < num_taps; k++) {
      int32_t idx = static_cast<int32_t>(n) - static_cast<int32_t>(k);
      if (idx >= 0) {
        sum += coeffs[k] * input[idx];
      }
    }
    output[n] = sum;
  }
}
