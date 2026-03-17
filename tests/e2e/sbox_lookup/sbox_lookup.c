#include <stdint.h>

__attribute__((noinline)) void
sbox_lookup(const uint32_t *restrict input_data,
            const uint32_t *restrict input_sbox,
            uint32_t *restrict output_result, uint32_t n) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0u; i < n; ++i) {
    uint32_t index = input_data[i] & 0xFFu;
    output_result[i] = input_sbox[index];
  }
}

int main(void) {
  uint32_t input_data[4] = {0u, 1u, 2u, 255u};
  uint32_t input_sbox[256];
  for (uint32_t i = 0u; i < 256u; ++i)
    input_sbox[i] = i;
  input_sbox[0] = 100u;
  input_sbox[1] = 200u;
  input_sbox[2] = 150u;
  input_sbox[255] = 255u;
  uint32_t output_result[4] = {0u, 0u, 0u, 0u};
  sbox_lookup(input_data, input_sbox, output_result, 4u);
  return (output_result[0] == 100u && output_result[1] == 200u &&
          output_result[2] == 150u && output_result[3] == 255u)
             ? 0
             : 1;
}
