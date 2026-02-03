//===- normalize.h - Normalization Graph Declarations -------------*- C++ -*-===//
//
// Composite kernel demonstrating fork-join pattern:
//   sum_kernel ---\
//                  --> scale_kernel
//   max_kernel ---/
//
// sum_kernel and max_kernel can execute in parallel (no data dependency).
// scale_kernel depends on both.
//
//===----------------------------------------------------------------------===//

#ifndef NORMALIZE_NORMALIZE_H
#define NORMALIZE_NORMALIZE_H

// Reference implementation
void normalize_cpu(const float *in, float *out, int n);

// Accelerated version with fork-join kernel graph
void normalize_dsa(const float *in, float *out, int n);

#endif // NORMALIZE_NORMALIZE_H
