//===- vecmul.h - Vector Multiplication Header -----------------------------===//
//
// Header for vector multiplication functions.
//
//===----------------------------------------------------------------------===//

#ifndef VECMUL_H
#define VECMUL_H

// CPU implementation of vector element-wise multiplication
void vecmul(const float *a, const float *b, float *c, int n);

// DSA-optimized version (for future loom target compilation)
void vecmul_dsa(const float *__restrict__ a, const float *__restrict__ b,
                float *__restrict__ c, int n);

#endif // VECMUL_H
