#include "scicomp_types.h"

void precond_diag(const float *r, const float *m_diag, float *z, int n) {
  if (!r || !m_diag || !z || n <= 0)
    return;
  for (int i = 0; i < n; ++i) {
    float denom = m_diag[i];
    z[i] = denom != 0.0f ? r[i] / denom : r[i];
  }
}

void precond_block4(const float *block4, const float *r, float *z, int n) {
  if (!block4 || !r || !z || n <= 0)
    return;

  int i = 0;
  for (; i + 3 < n; i += 4) {
    for (int row = 0; row < 4; ++row) {
      float sum = 0.0f;
      for (int col = 0; col < 4; ++col)
        sum += block4[row * 4 + col] * r[i + col];
      z[i + row] = sum;
    }
  }
  for (; i < n; ++i)
    z[i] = r[i];
}
