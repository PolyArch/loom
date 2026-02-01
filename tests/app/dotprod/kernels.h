//===- kernels.h - Dot Product Kernel Declarations ----------------*- C++ -*-===//
//
// Leaf kernels for dot product computation: vecmul and vecsum.
// These are the atomic building blocks of the dotprod graph.
//
//===----------------------------------------------------------------------===//

#ifndef DOTPROD_KERNELS_H
#define DOTPROD_KERNELS_H

// Element-wise vector multiplication
// Parallel kernel with 8-way unrolling
void vecmul_kernel(const float *a, const float *b, float *products, int n);

// Vector sum reduction
// Parallel kernel with 4-way unrolling
void vecsum_kernel(const float *data, float *result, int n);

#endif // DOTPROD_KERNELS_H
