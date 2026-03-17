#include <stdint.h>

int32_t vecsum(int32_t *restrict a, int32_t init, int32_t n) {
  int32_t sum = init;
  for (int32_t i = 0; i < n; ++i)
    sum += a[i];
  return sum;
}

int main(void) {
  int32_t a[4] = {1, 2, 3, 4};
  return vecsum(a, 5, 4) == 15 ? 0 : 1;
}
