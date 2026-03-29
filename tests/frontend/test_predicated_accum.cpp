// T14: Predicated accumulation.
// Tests: conditional accumulation pattern with select + carry
#include <cstdint>

int32_t predicated_sum(const int32_t *a, const int32_t *valid, int n) {
  int32_t sum = 0;
  for (int i = 0; i < n; ++i) {
    if (valid[i])
      sum += a[i];
  }
  return sum;
}

int main() {
  constexpr int n = 8;
  int32_t a[n] = {10, 20, 30, 40, 50, 60, 70, 80};
  int32_t valid[n] = {1, 0, 1, 0, 1, 0, 1, 0};

  int32_t result = predicated_sum(a, valid, n);
  (void)result;
  return 0;
}
