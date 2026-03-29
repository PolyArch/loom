// T13: Min reduction recognized.
// Tests: loop-carried min accumulation via compare+select
#include <cstdint>

int32_t min_reduce(const int32_t *a, int n) {
  int32_t min_val = a[0];
  for (int i = 1; i < n; ++i) {
    if (a[i] < min_val)
      min_val = a[i];
  }
  return min_val;
}

int main() {
  constexpr int n = 8;
  int32_t a[n] = {5, 3, 8, 1, 9, 2, 7, 4};

  int32_t result = min_reduce(a, n);
  (void)result;
  return 0;
}
