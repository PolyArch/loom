// Loom app test driver: Autocorrelation
#include "autocorrelation.h"
#include <cmath>
#include <cstdio>

int main() {
  const uint32_t x_size = 128;
  const uint32_t max_lag = 32;

  // Input signal
  float x[x_size];

  // Output arrays
  float expect_autocorr[max_lag];
  float calculated_autocorr[max_lag];

  // Initialize signal
  for (uint32_t i = 0; i < x_size; i++) {
    x[i] = sinf(2.0f * 3.14159f * i / 16.0f);
  }

  // Test auto-correlation
  autocorrelation_cpu(x, expect_autocorr, x_size, max_lag);
  autocorrelation_dsa(x, calculated_autocorr, x_size, max_lag);

  // Verify results
  bool passed = true;
  for (uint32_t i = 0; i < max_lag; i++) {
    if (fabsf(expect_autocorr[i] - calculated_autocorr[i]) > 1e-4f) {
      printf("Mismatch at lag %u: expected %f, got %f\n", i, expect_autocorr[i],
             calculated_autocorr[i]);
      passed = false;
    }
  }

  if (passed) {
    printf("Autocorrelation: PASSED\n");
  } else {
    printf("Autocorrelation: FAILED\n");
  }

  return passed ? 0 : 1;
}
