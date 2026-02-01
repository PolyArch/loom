//===- kernels.cpp - Leaf Kernel Implementations ------------------*- C++
//-*-===//
//
// Leaf kernels for dot product: vecmul_kernel and vecsum_kernel.
// Each kernel is annotated with LOOM_ACCEL and parallelism hints.
//
//===----------------------------------------------------------------------===//

#include "kernels.h"
#include <loom/loom.h>

//===----------------------------------------------------------------------===//
// vecmul_kernel: Element-wise multiplication
// Graph role: Producer - writes to 'products' buffer
//===----------------------------------------------------------------------===//

LOOM_TARGET("temporal")
LOOM_ACCEL("vecmul")
void vecmul_kernel(const float *__restrict__ a, const float *__restrict__ b,
                   float *__restrict__ products, int n) {
  LOOM_PARALLEL(4)
  for (int i = 0; i < n; ++i) {
    products[i] = a[i] * b[i];
  }
}

//===----------------------------------------------------------------------===//
// vecsum_kernel: Reduction sum
// Graph role: Consumer - reads from input buffer, produces scalar result
//===----------------------------------------------------------------------===//

LOOM_TARGET("temporal")
LOOM_ACCEL("vecsum")
void vecsum_kernel(const float *__restrict__ data, float *__restrict__ result,
                   int n) {
  LOOM_REDUCE(+)
  float sum = 0.0f;
  LOOM_PARALLEL(4)
  for (int i = 0; i < n; ++i) {
    sum += data[i];
  }
  *result = sum;
}
