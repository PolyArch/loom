#include <stdint.h>

void cumsum(int32_t *restrict input, int32_t *restrict output, int32_t n) {
  int32_t sum = 0;
  for (int32_t i = 0; i < n; ++i) {
    sum += input[i];
    output[i] = sum;
  }
}

int main(void) {
  int32_t input[4] = {1, 2, 3, 4};
  int32_t output[4] = {0, 0, 0, 0};
  cumsum(input, output, 4);
  return (output[0] == 1 && output[1] == 3 &&
          output[2] == 6 && output[3] == 10)
             ? 0
             : 1;
}
