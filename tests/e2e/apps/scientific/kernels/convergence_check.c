#include "scicomp_types.h"

#include <math.h>

float conv_l2(const float *r, int n) {
  if (!r || n <= 0)
    return 0.0f;
  float sum = 0.0f;
  for (int i = 0; i < n; ++i)
    sum += r[i] * r[i];
  return sqrtf(sum);
}

float conv_max(const float *r, int n) {
  if (!r || n <= 0)
    return 0.0f;
  float max_val = 0.0f;
  for (int i = 0; i < n; ++i) {
    float v = r[i];
    if (v < 0.0f)
      v = -v;
    if (v > max_val)
      max_val = v;
  }
  return max_val;
}
