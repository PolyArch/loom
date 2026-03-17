#include <stdint.h>

void parity(const uint32_t *restrict input_data,
            uint32_t *restrict output_parity, uint32_t n) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t value = input_data[i];
    uint32_t result = 0;

    while (value != 0) {
      result ^= (value & 1u);
      value >>= 1;
    }

    output_parity[i] = result;
  }
}

int main(void) {
  uint32_t input_data[4] = {7u, 6u, 14u, 11u};
  uint32_t output_parity[4] = {0u, 0u, 0u, 0u};
  parity(input_data, output_parity, 4u);
  return (output_parity[0] == 1u && output_parity[1] == 0u &&
          output_parity[2] == 1u && output_parity[3] == 1u)
             ? 0
             : 1;
}
