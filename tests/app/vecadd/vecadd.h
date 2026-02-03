//===- vecadd.h - Vector Addition Header -----------------------------------===//
//
// Header for vector addition functions.
//
//===----------------------------------------------------------------------===//

#ifndef VECADD_H
#define VECADD_H

// CPU implementation of vector addition
void vecadd_cpu(const float *a, const float *b, float *c, int n);

// Accelerated version (for future loom target compilation)
void vecadd_dsa(const float *__restrict__ a, const float *__restrict__ b,
                float *__restrict__ c, int n);

#endif // VECADD_H
