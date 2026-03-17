#include <stdint.h>

void rotate_bits(const uint32_t *restrict input_data,
                 const uint32_t *restrict input_shift,
                 uint32_t *restrict output_result, uint32_t n) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t value = input_data[i];
    uint32_t shift = input_shift[i] & 31u;
    output_result[i] =
        (shift == 0u) ? value : ((value << shift) | (value >> (32u - shift)));
  }
}

int main(void) {
  uint32_t input_data[4] = {0xF0F0F0F0u, 8u, 0xAAAAAAAAu, 0x12345678u};
  uint32_t input_shift[4] = {4u, 1u, 8u, 0u};
  uint32_t output_result[4] = {0u, 0u, 0u, 0u};
  rotate_bits(input_data, input_shift, output_result, 4u);
  return (output_result[0] == 0x0F0F0F0Fu && output_result[1] == 16u &&
          output_result[2] == 0xAAAAAAAau && output_result[3] == 0x12345678u)
             ? 0
             : 1;
}
