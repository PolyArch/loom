#include <stdint.h>

int32_t pairsum_tagged_temporal(int32_t *restrict a, int32_t n) {
  int32_t sum = 0;
  for (int32_t i = 0; i + 1 < n; ++i)
    sum += a[i] + a[i + 1];
  return sum;
}

int main(void) {
  int32_t a[5] = {1, 2, 3, 4, 5};
  return pairsum_tagged_temporal(a, 5) == 24 ? 0 : 1;
}
