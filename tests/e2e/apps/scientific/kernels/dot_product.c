#include "scicomp_types.h"

float dot_sequential(const float *a, const float *b, int n) {
  if (!a || !b || n <= 0)
    return 0.0f;

  float sum = 0.0f;
  for (int i = 0; i < n; ++i)
    sum += a[i] * b[i];
  return sum;
}

float dot_tree_reduce(const float *a, const float *b, int n) {
  if (!a || !b || n <= 0)
    return 0.0f;

  float p0 = 0.0f, p1 = 0.0f, p2 = 0.0f, p3 = 0.0f;
  int i = 0;
  for (; i + 3 < n; i += 4) {
    p0 += a[i] * b[i];
    p1 += a[i + 1] * b[i + 1];
    p2 += a[i + 2] * b[i + 2];
    p3 += a[i + 3] * b[i + 3];
  }
  for (; i < n; ++i)
    p0 += a[i] * b[i];
  return (p0 + p1) + (p2 + p3);
}

float dot_sequential_unroll2(const float *a, const float *b, int n) {
  if (!a || !b || n <= 0)
    return 0.0f;

  float sum0 = 0.0f;
  float sum1 = 0.0f;
  int i = 0;
  for (; i + 1 < n; i += 2) {
    sum0 += a[i] * b[i];
    sum1 += a[i + 1] * b[i + 1];
  }
  for (; i < n; ++i)
    sum0 += a[i] * b[i];
  return sum0 + sum1;
}
