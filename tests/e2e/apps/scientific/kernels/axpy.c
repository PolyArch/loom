#include "scicomp_types.h"

void axpy_basic(float alpha, const float *x, float *y, int n) {
  if (!x || !y || n <= 0)
    return;
  for (int i = 0; i < n; ++i)
    y[i] = alpha * x[i] + y[i];
}

void axpy_unroll4(float alpha, const float *x, float *y, int n) {
  if (!x || !y || n <= 0)
    return;

  int i = 0;
  for (; i + 3 < n; i += 4) {
    y[i] = alpha * x[i] + y[i];
    y[i + 1] = alpha * x[i + 1] + y[i + 1];
    y[i + 2] = alpha * x[i + 2] + y[i + 2];
    y[i + 3] = alpha * x[i + 3] + y[i + 3];
  }
  for (; i < n; ++i)
    y[i] = alpha * x[i] + y[i];
}
