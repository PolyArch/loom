#include <stdint.h>

void vecscale(int32_t *restrict a, int32_t alpha, int32_t *restrict b,
              int32_t n) {
  for (int32_t i = 0; i < n; ++i)
    b[i] = alpha * a[i];
}

int main(void) {
  int32_t a[4] = {2, 4, 6, 8};
  int32_t b[4] = {0, 0, 0, 0};

  vecscale(a, 3, b, 4);
  return (b[0] == 6 && b[1] == 12 && b[2] == 18 && b[3] == 24) ? 0 : 1;
}
