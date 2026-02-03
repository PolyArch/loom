//===- normalize.cpp - Normalization Composite Kernel -------------*- C++
//-*-===//
//
// Demonstrates fork-join pattern in kernel graph.
//
// Kernel Graph (fork-join):
//
//        in
//       /   \
//      v     v
//   [sum]   [max]    <-- parallel (no dependency)
//      \     /
//       v   v
//      [scale]       <-- depends on both sum and max
//         |
//         v
//        out
//
// In this example, we only use 'sum' for normalization, but 'max' is
// computed in parallel to demonstrate the fork-join pattern.
//
//===----------------------------------------------------------------------===//

#include "normalize.h"
#include "kernels.h"
#include <loom/loom.h>

//===----------------------------------------------------------------------===//
// Reference implementation
//===----------------------------------------------------------------------===//

void normalize_cpu(const float *in, float *out, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    sum += in[i];
  }
  float scale = (sum > 0.0f) ? (1.0f / sum) : 1.0f;
  for (int i = 0; i < n; ++i) {
    out[i] = in[i] * scale;
  }
}

//===----------------------------------------------------------------------===//
// Accelerated version with fork-join kernel graph
//===----------------------------------------------------------------------===//

LOOM_TARGET("temporal")
LOOM_ACCEL("normalize")
void normalize_dsa(const float *__restrict__ in, float *__restrict__ out,
                   int n) {
  // Fork: sum_kernel and max_kernel read from 'in' with no dependency
  float sum_result;
  float max_result;

  // These two kernels can execute in parallel
  sum_kernel(in, &sum_result, n); // Reads: in, Writes: sum_result
  max_kernel(in, &max_result, n); // Reads: in, Writes: max_result

  // Join: scale_kernel depends on sum_kernel (uses sum_result)
  // (max_result computed for demonstration, could be used for range scaling)
  scale_kernel(in, sum_result, out, n); // Reads: in, sum_result, Writes: out

  // max_result available here for future use
  (void)max_result;
}
