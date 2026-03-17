#include <stdint.h>

void vecmul(int32_t *restrict a, int32_t *restrict b, int32_t *restrict c,
            int32_t n) {
  for (int32_t i = 0; i < n; ++i)
    c[i] = a[i] * b[i];
}

int main(void) {
  int32_t a[4] = {1, 2, 3, 4};
  int32_t b[4] = {5, 6, 7, 8};
  int32_t c[4] = {0, 0, 0, 0};

  vecmul(a, b, c, 4);
  return (c[0] == 5 && c[1] == 12 && c[2] == 21 && c[3] == 32) ? 0 : 1;
}
