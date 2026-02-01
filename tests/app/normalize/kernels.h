//===- kernels.h - Normalization Kernel Declarations --------------*- C++ -*-===//
//
// Leaf kernels for vector normalization: sum, max, scale.
// Demonstrates fork-join pattern in kernel graph.
//
//===----------------------------------------------------------------------===//

#ifndef NORMALIZE_KERNELS_H
#define NORMALIZE_KERNELS_H

// Compute sum of all elements (reduction)
void sum_kernel(const float *data, float *result, int n);

// Compute max of all elements (reduction)
void max_kernel(const float *data, float *result, int n);

// Scale each element: out[i] = (in[i] - min) / (max - min)
// For simplicity, we use: out[i] = in[i] / sum
void scale_kernel(const float *in, float sum, float *out, int n);

#endif // NORMALIZE_KERNELS_H
