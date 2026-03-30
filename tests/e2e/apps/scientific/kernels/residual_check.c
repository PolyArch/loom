#include "scicomp_types.h"

#include <math.h>

float residual_l2(const float *curr, const float *prev, int rows, int cols) {
  if (!curr || !prev || rows <= 0 || cols <= 0)
    return 0.0f;

  float sum = 0.0f;
  for (int i = 0; i < rows * cols; ++i) {
    float diff = curr[i] - prev[i];
    sum += diff * diff;
  }
  return sqrtf(sum);
}

float residual_max(const float *curr, const float *prev, int rows, int cols) {
  if (!curr || !prev || rows <= 0 || cols <= 0)
    return 0.0f;

  float max_val = 0.0f;
  for (int i = 0; i < rows * cols; ++i) {
    float diff = curr[i] - prev[i];
    if (diff < 0.0f)
      diff = -diff;
    if (diff > max_val)
      max_val = diff;
  }
  return max_val;
}
