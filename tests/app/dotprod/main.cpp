//===- main.cpp - Dot Product Test Driver --------------------------*- C++ -*-===//
//
// Tests the dot product implementation with known values.
//
//===----------------------------------------------------------------------===//

#include "dotprod.h"
#include <cmath>
#include <cstdio>

constexpr int N = 8;

int main() {
  float a[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float b[N] = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f};

  // Expected: 1*0.5 + 2*1 + 3*1.5 + 4*2 + 5*2.5 + 6*3 + 7*3.5 + 8*4
  //         = 0.5 + 2 + 4.5 + 8 + 12.5 + 18 + 24.5 + 32 = 102.0
  float expected = 102.0f;

  // Test reference implementation
  float ref_result = dotprod(a, b, N);

  // Test DSA implementation
  float dsa_result = dotprod_dsa(a, b, N);

  printf("Dot Product Results:\n");
  printf("a = [");
  for (int i = 0; i < N; ++i) {
    printf("%.1f%s", a[i], i < N - 1 ? ", " : "");
  }
  printf("]\n");

  printf("b = [");
  for (int i = 0; i < N; ++i) {
    printf("%.1f%s", b[i], i < N - 1 ? ", " : "");
  }
  printf("]\n");

  printf("a . b = %.1f (expected %.1f)\n", dsa_result, expected);

  // Verify results
  constexpr float epsilon = 0.001f;
  bool ref_ok = std::fabs(ref_result - expected) < epsilon;
  bool dsa_ok = std::fabs(dsa_result - expected) < epsilon;

  if (ref_ok && dsa_ok) {
    printf("PASSED: All results correct!\n");
    return 0;
  } else {
    if (!ref_ok) {
      printf("FAILED: Reference result %.1f != expected %.1f\n", ref_result,
             expected);
    }
    if (!dsa_ok) {
      printf("FAILED: DSA result %.1f != expected %.1f\n", dsa_result,
             expected);
    }
    return 1;
  }
}
