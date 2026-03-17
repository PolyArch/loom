#include <stdint.h>

void gf_mul(const uint32_t *restrict input_a, const uint32_t *restrict input_b,
            uint32_t *restrict output_c, uint32_t n) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t a = input_a[i] & 0xFFu;
    uint32_t b = input_b[i] & 0xFFu;
    uint32_t p = 0;

#pragma clang loop vectorize(disable) interleave(disable)
    for (uint32_t j = 0; j < 8; ++j) {
      if (b & 1u)
        p ^= a;
      uint32_t hi_bit_set = a & 0x80u;
      a <<= 1u;
      if (hi_bit_set)
        a ^= 0x1Bu;
      b >>= 1u;
    }

    output_c[i] = p & 0xFFu;
  }
}

int main(void) {
  uint32_t input_a[3] = {83, 202, 1};
  uint32_t input_b[3] = {202, 83, 5};
  uint32_t output_c[3] = {0, 0, 0};
  gf_mul(input_a, input_b, output_c, 3);
  return (output_c[0] == 1 && output_c[1] == 1 && output_c[2] == 5) ? 0 : 1;
}
