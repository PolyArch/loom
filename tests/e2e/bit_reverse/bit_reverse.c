#include <stdint.h>

void bit_reverse(const uint32_t *restrict input_data,
                 uint32_t *restrict output_reversed, uint32_t n) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t value = input_data[i];
    uint32_t result = 0;

#pragma clang loop vectorize(disable) interleave(disable)
    for (uint32_t bit = 0; bit < 32u; ++bit) {
      result = (result << 1) | (value & 1u);
      value >>= 1;
    }

    output_reversed[i] = result;
  }
}

int main(void) {
  uint32_t input_data[4] = {0x12345678u, 0xABCDEF00u, 1u, 0x80000000u};
  uint32_t output_reversed[4] = {0u, 0u, 0u, 0u};
  bit_reverse(input_data, output_reversed, 4u);
  return (output_reversed[0] == 0x1E6A2C48u &&
          output_reversed[1] == 0x00F7B3D5u &&
          output_reversed[2] == 0x80000000u &&
          output_reversed[3] == 1u)
             ? 0
             : 1;
}
