//===- kernels.cpp - Normalization Leaf Kernels -------------------*- C++
//-*-===//
//
// Leaf kernels for normalization: sum_kernel, max_kernel, scale_kernel.
//
//===----------------------------------------------------------------------===//

#include "kernels.h"
#include <loom/loom.h>

//===----------------------------------------------------------------------===//
// sum_kernel: Reduction sum
//===----------------------------------------------------------------------===//

LOOM_TARGET("temporal")
LOOM_ACCEL("sum")
void sum_kernel(const float *__restrict__ data, float *__restrict__ result,
                int n) {
  LOOM_REDUCE(+)
  float sum = 0.0f;
  LOOM_PARALLEL(4)
  for (int i = 0; i < n; ++i) {
    sum += data[i];
  }
  *result = sum;
}

//===----------------------------------------------------------------------===//
// max_kernel: Reduction max
//===----------------------------------------------------------------------===//

LOOM_TARGET("temporal")
LOOM_ACCEL("max")
void max_kernel(const float *__restrict__ data, float *__restrict__ result,
                int n) {
  LOOM_REDUCE(max)
  float maxval = data[0];
  LOOM_PARALLEL(4)
  for (int i = 1; i < n; ++i) {
    if (data[i] > maxval) {
      maxval = data[i];
    }
  }
  *result = maxval;
}

//===----------------------------------------------------------------------===//
// scale_kernel: Element-wise scaling
//===----------------------------------------------------------------------===//

LOOM_TARGET("temporal")
LOOM_ACCEL("scale")
void scale_kernel(const float *__restrict__ in, float sum,
                  float *__restrict__ out, int n) {
  float scale = (sum > 0.0f) ? (1.0f / sum) : 1.0f;
  LOOM_PARALLEL(4)
  for (int i = 0; i < n; ++i) {
    out[i] = in[i] * scale;
  }
}
