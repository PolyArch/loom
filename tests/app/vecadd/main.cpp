//===- main.cpp - Vector Addition Test Program -----------------------------===//
//
// Test program for loom compiler - compiles and runs vector addition.
//
//===----------------------------------------------------------------------===//

#include "vecadd.h"
#include <cmath>
#include <cstdio>

int main() {
  const int N = 8;
  float a[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float b[N] = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f};
  float c_cpu[N] = {0};
  float c_accel[N] = {0};

  // Run CPU and accelerated versions
  vecadd_cpu(a, b, c_cpu, N);
  vecadd_dsa(a, b, c_accel, N);

  // Print results
  printf("Vector Addition Results:\n");
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

  printf("c = a + b = [");
  for (int i = 0; i < N; ++i) {
    printf("%.1f%s", c_accel[i], i < N - 1 ? ", " : "");
  }
  printf("]\n");

  // Verify results
  bool passed = true;
  for (int i = 0; i < N; ++i) {
    float expected = a[i] + b[i];
    if (std::fabs(c_cpu[i] - expected) > 1e-5f ||
        std::fabs(c_accel[i] - expected) > 1e-5f) {
      printf("ERROR: c[%d] = %.1f (cpu) %.1f (accel), expected %.1f\n", i,
             c_cpu[i], c_accel[i], expected);
      passed = false;
    }
  }

  if (passed) {
    printf("PASSED: All results correct!\n");
    return 0;
  } else {
    printf("FAILED: Some results incorrect!\n");
    return 1;
  }
}
