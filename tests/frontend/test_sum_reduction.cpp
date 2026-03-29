// T12: Sum reduction recognized.
// Tests: loop-carried sum accumulation annotated with reduction kind
#include <cstdint>

int32_t sum_reduce(const int32_t *a, int n) {
  int32_t sum = 0;
  for (int i = 0; i < n; ++i)
    sum += a[i];
  return sum;
}

int main() {
  constexpr int n = 8;
  int32_t a[n] = {1, 2, 3, 4, 5, 6, 7, 8};

  int32_t result = sum_reduce(a, n);
  (void)result;
  return 0;
}
