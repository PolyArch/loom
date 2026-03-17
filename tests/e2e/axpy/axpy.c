#include <stdint.h>

void axpy(int32_t *restrict x, int32_t *restrict y, int32_t *restrict out,
          int32_t alpha, int32_t n) {
  for (int32_t i = 0; i < n; ++i)
    out[i] = alpha * x[i] + y[i];
}

int main(void) {
  int32_t x[4] = {1, 2, 3, 4};
  int32_t y[4] = {10, 20, 30, 40};
  int32_t out[4] = {0, 0, 0, 0};

  axpy(x, y, out, 3, 4);
  return (out[0] == 13 && out[1] == 26 && out[2] == 39 && out[3] == 52) ? 0
                                                                          : 1;
}
