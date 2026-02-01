//===- vecadd.cpp - Vector Addition Implementation
//-------------------------===//
//
// Simple vector addition for testing loom compiler.
//
//===----------------------------------------------------------------------===//

#include "vecadd.h"
#include <loom/loom.h>

void vecadd(const float *a, const float *b, float *c, int n) {
  for (int i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
}

// DSA-optimized version using loom pragma
LOOM_ACCEL()
void vecadd_dsa(const float *__restrict__ a, const float *__restrict__ b,
                float *__restrict__ c, int n) {
  LOOM_PARALLEL(4, CONTIGUOUS)
  LOOM_TRIPCOUNT(256)
  for (int i = 0; i < n; ++i) {
    c[i] = a[i] + b[i];
  }
}
