// FIR Filter test driver

#include "fir_filter.h"
#include <cmath>
#include <cstdio>

int main() {
  const uint32_t input_size = 8;
  const uint32_t num_taps = 3;

  // Input signal
  float input[input_size] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};

  // Filter coefficients (simple averaging filter)
  float coeffs[num_taps] = {0.25f, 0.5f, 0.25f};

  // Output arrays
  float cpu_output[input_size];
  float dsa_output[input_size];

  // Run both implementations
  fir_filter_cpu(input, coeffs, cpu_output, input_size, num_taps);
  fir_filter_dsa(input, coeffs, dsa_output, input_size, num_taps);

  // Print results
  printf("FIR Filter Results (3-tap averaging filter):\n");
  printf("Input:      [");
  for (uint32_t i = 0; i < input_size; i++) {
    printf("%.1f%s", input[i], i < input_size - 1 ? ", " : "");
  }
  printf("]\n");

  printf("Coeffs:     [");
  for (uint32_t i = 0; i < num_taps; i++) {
    printf("%.2f%s", coeffs[i], i < num_taps - 1 ? ", " : "");
  }
  printf("]\n");

  printf("CPU Output: [");
  for (uint32_t i = 0; i < input_size; i++) {
    printf("%.2f%s", cpu_output[i], i < input_size - 1 ? ", " : "");
  }
  printf("]\n");

  printf("DSA Output: [");
  for (uint32_t i = 0; i < input_size; i++) {
    printf("%.2f%s", dsa_output[i], i < input_size - 1 ? ", " : "");
  }
  printf("]\n");

  // Verify results
  bool passed = true;
  for (uint32_t i = 0; i < input_size; i++) {
    if (fabsf(cpu_output[i] - dsa_output[i]) > 1e-5f) {
      passed = false;
      break;
    }
  }

  if (passed) {
    printf("PASSED: All results correct!\n");
    return 0;
  } else {
    printf("FAILED: Results mismatch!\n");
    return 1;
  }
}
