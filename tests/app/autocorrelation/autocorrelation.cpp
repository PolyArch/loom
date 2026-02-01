// Loom kernel implementation
#include "autocorrelation.h"
#include "loom/loom.h"

// CPU implementation of auto-correlation
void autocorrelation_cpu(const float *__restrict__ x, float *__restrict__ output,
                         uint32_t x_size, uint32_t max_lag) {
  for (uint32_t lag = 0; lag < max_lag; lag++) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < x_size - lag; i++) {
      sum += x[i] * x[i + lag];
    }
    output[lag] = sum;
  }
}

// Accelerator implementation of auto-correlation
LOOM_ACCEL()
void autocorrelation_dsa(const float *__restrict__ x, float *__restrict__ output,
                         uint32_t x_size, uint32_t max_lag) {
  LOOM_PARALLEL(4, contiguous)
  LOOM_TRIPCOUNT_FULL(256, 256, 1, 1024)
  for (uint32_t lag = 0; lag < max_lag; lag++) {
    float sum = 0.0f;
    for (uint32_t i = 0; i < x_size - lag; i++) {
      sum += x[i] * x[i + lag];
    }
    output[lag] = sum;
  }
}
