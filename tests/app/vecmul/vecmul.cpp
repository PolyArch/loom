//===- vecmul.cpp - Vector Multiplication Implementation
//-------------------===//
//
// Simple vector element-wise multiplication for testing loom compiler.
//
//===----------------------------------------------------------------------===//

#include "vecmul.h"
#include <loom/loom.h>

void vecmul_cpu(const float *a, const float *b, float *c, int n) {
  for (int i = 0; i < n; ++i) {
    c[i] = a[i] * b[i];
  }
}

// Accelerated version using loom pragma
LOOM_ACCEL()
void vecmul_dsa(const float *__restrict__ a, const float *__restrict__ b,
                float *__restrict__ c, int n) {
  LOOM_PARALLEL(4, contiguous)
  LOOM_TRIPCOUNT_FULL(256, 256, 1, 1024)
  for (int i = 0; i < n; ++i) {
    c[i] = a[i] * b[i];
  }
}
