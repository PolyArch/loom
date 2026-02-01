// Loom kernel: Autocorrelation
#ifndef AUTOCORRELATION_H
#define AUTOCORRELATION_H

#include <cstdint>

// CPU implementation of auto-correlation
void autocorrelation_cpu(const float *__restrict__ x, float *__restrict__ output,
                         uint32_t x_size, uint32_t max_lag);

// Accelerator implementation of auto-correlation
void autocorrelation_dsa(const float *__restrict__ x, float *__restrict__ output,
                         uint32_t x_size, uint32_t max_lag);

#endif // AUTOCORRELATION_H
