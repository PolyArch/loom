//===- dotprod.cpp - Dot Product Composite Kernel -----------------*- C++
//-*-===//
//
// Composite kernel implementing dot product as a 2-node kernel graph:
//   vecmul_kernel -> vecsum_kernel
//
// The intermediate products buffer creates an edge in the kernel graph
// connecting the producer (vecmul) to the consumer (vecsum).
//
//===----------------------------------------------------------------------===//

#include "dotprod.h"
#include "kernels.h"
#include <loom/loom.h>

//===----------------------------------------------------------------------===//
// Reference implementation (no DSA)
//===----------------------------------------------------------------------===//

float dotprod(const float *a, const float *b, int n) {
  float sum = 0.0f;
  for (int i = 0; i < n; ++i) {
    sum += a[i] * b[i];
  }
  return sum;
}

//===----------------------------------------------------------------------===//
// DSA-optimized composite kernel
//
// Kernel Graph:
//   [vecmul_kernel] --products--> [vecsum_kernel]
//
// The compiler infers:
//   - vecmul_kernel produces 'products' buffer
//   - vecsum_kernel consumes 'products' buffer
//   - Therefore: vecsum_kernel depends on vecmul_kernel
//===----------------------------------------------------------------------===//

LOOM_TARGET("temporal")
LOOM_ACCEL("dotprod")
float dotprod_dsa(const float *__restrict__ a, const float *__restrict__ b,
                  int n) {
  // Intermediate buffer - becomes an edge in the kernel graph
  float products[1024]; // Fixed size for testing

  float result;

  // Node 1: Element-wise multiply (producer)
  vecmul_kernel(a, b, products, n);

  // Node 2: Reduction sum (consumer) - depends on Node 1
  vecsum_kernel(products, &result, n);

  return result;
}
