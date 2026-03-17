#include <stdint.h>

void xor_block(const uint32_t *restrict input_a,
               const uint32_t *restrict input_b,
               uint32_t *restrict output_c, uint32_t n) {
  for (uint32_t i = 0; i < n; ++i)
    output_c[i] = input_a[i] ^ input_b[i];
}

int main(void) {
  uint32_t input_a[4] = {1, 3, 5, 7};
  uint32_t input_b[4] = {2, 4, 6, 8};
  uint32_t output_c[4] = {0, 0, 0, 0};
  xor_block(input_a, input_b, output_c, 4);
  return (output_c[0] == 3 && output_c[1] == 7 && output_c[2] == 3 &&
          output_c[3] == 15)
             ? 0
             : 1;
}
