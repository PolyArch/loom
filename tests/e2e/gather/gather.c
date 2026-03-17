#include <stdint.h>

void gather(int32_t *restrict src, int32_t *restrict indices,
            int32_t *restrict dst, int32_t n, int32_t srcSize) {
  for (int32_t i = 0; i < n; ++i) {
    int32_t idx = indices[i];
    if (idx < srcSize)
      dst[i] = src[idx];
    else
      dst[i] = 0;
  }
}

int main(void) {
  int32_t src[6] = {10, 20, 30, 40, 50, 60};
  int32_t indices[4] = {0, 2, 5, 9};
  int32_t dst[4] = {0, 0, 0, 0};

  gather(src, indices, dst, 4, 6);
  return (dst[0] == 10 && dst[1] == 30 && dst[2] == 60 && dst[3] == 0) ? 0
                                                                         : 1;
}
