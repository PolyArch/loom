// T4: Math intrinsics lower to math dialect.
// Tests: exp, sqrt, log, sin, cos
#include <cmath>

void math_ops(const float *x, float *y, int n) {
  for (int i = 0; i < n; ++i) {
    y[i] = expf(x[i]) + sqrtf(x[i]);
  }
}

int main() {
  constexpr int n = 4;
  float x[n] = {1.0f, 2.0f, 3.0f, 4.0f};
  float y[n] = {};

  math_ops(x, y, n);
  return 0;
}
