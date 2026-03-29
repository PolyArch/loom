// T8: Simple scf.if converted to arith.select.
// Tests: absolute value pattern uses select instead of gate/carry
#include <cstdint>

void absolute_value(const int32_t *x, int32_t *result, int n) {
  for (int i = 0; i < n; ++i) {
    // Ternary: r = (x > 0) ? x : -x
    result[i] = (x[i] > 0) ? x[i] : -x[i];
  }
}

int main() {
  constexpr int n = 4;
  int32_t x[n] = {5, -3, 0, -7};
  int32_t result[n] = {};

  absolute_value(x, result, n);
  return 0;
}
