//===- dotprod.h - Dot Product Graph Declarations -----------------*- C++ -*-===//
//
// Composite kernel that computes dot product using a kernel graph:
//   dotprod = vecmul_kernel -> vecsum_kernel
//
//===----------------------------------------------------------------------===//

#ifndef DOTPROD_DOTPROD_H
#define DOTPROD_DOTPROD_H

// Reference implementation
float dotprod_cpu(const float *a, const float *b, int n);

// Accelerated composite kernel
// Contains a 2-node kernel graph: vecmul -> vecsum
float dotprod_dsa(const float *a, const float *b, int n);

#endif // DOTPROD_DOTPROD_H
