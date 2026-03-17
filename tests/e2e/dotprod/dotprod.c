#include <stdint.h>

int32_t dotprod(int32_t *restrict a, int32_t *restrict b, int32_t n) {
  int32_t sum = 0;
  for (int32_t i = 0; i < n; ++i)
    sum += a[i] * b[i];
  return sum;
}

int main(void) {
  int32_t a[4] = {1, 2, 3, 4};
  int32_t b[4] = {5, 6, 7, 8};
  return dotprod(a, b, 4) == 70 ? 0 : 1;
}
