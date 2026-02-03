//===- main.cpp - Normalization Test Driver -----------------------*- C++ -*-===//
//
// Tests the normalization implementation with known values.
//
//===----------------------------------------------------------------------===//

#include "normalize.h"
#include <cmath>
#include <cstdio>

constexpr int N = 8;

int main() {
  float in[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float out_ref[N];
  float out_accel[N];

  // Sum = 1+2+3+4+5+6+7+8 = 36
  // Expected normalized: each element / 36
  float expected[N];
  float sum = 36.0f;
  for (int i = 0; i < N; ++i) {
    expected[i] = in[i] / sum;
  }

  // Run reference
  normalize_cpu(in, out_ref, N);

  // Run accelerated version
  normalize_dsa(in, out_accel, N);

  printf("Normalization Results (fork-join graph):\n");
  printf("in  = [");
  for (int i = 0; i < N; ++i) {
    printf("%.1f%s", in[i], i < N - 1 ? ", " : "");
  }
  printf("]\n");

  printf("out = [");
  for (int i = 0; i < N; ++i) {
    printf("%.4f%s", out_accel[i], i < N - 1 ? ", " : "");
  }
  printf("]\n");

  printf("sum = %.1f\n", sum);

  // Verify
  constexpr float epsilon = 0.0001f;
  bool all_ok = true;
  for (int i = 0; i < N; ++i) {
    if (std::fabs(out_ref[i] - expected[i]) > epsilon ||
        std::fabs(out_accel[i] - expected[i]) > epsilon) {
      all_ok = false;
      printf("FAILED at index %d: ref=%.4f, accel=%.4f, expected=%.4f\n", i,
             out_ref[i], out_accel[i], expected[i]);
    }
  }

  if (all_ok) {
    printf("PASSED: All results correct!\n");
    return 0;
  }
  return 1;
}
