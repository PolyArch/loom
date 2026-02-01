//===- main.cpp - Vector Addition Test Program -----------------------------===//
//
// Test program for loom compiler - compiles and runs vector addition.
//
//===----------------------------------------------------------------------===//

#include "vecadd.h"
#include <cstdio>

int main() {
  const int N = 8;
  float a[N] = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f, 8.0f};
  float b[N] = {0.5f, 1.0f, 1.5f, 2.0f, 2.5f, 3.0f, 3.5f, 4.0f};
  float c[N] = {0};

  // Run vector addition
  vecadd(a, b, c, N);

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
    printf("%.1f%s", c[i], i < N - 1 ? ", " : "");
  }
  printf("]\n");

  // Verify results
  bool passed = true;
  for (int i = 0; i < N; ++i) {
    float expected = a[i] + b[i];
    if (c[i] != expected) {
      printf("ERROR: c[%d] = %.1f, expected %.1f\n", i, c[i], expected);
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
