#include <stdint.h>

void find_first_set(const uint32_t *restrict input_data,
                    uint32_t *restrict output_position, uint32_t n) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t value = input_data[i];

    if (value == 0u) {
      output_position[i] = 0u;
    } else {
      uint32_t position = 1u;
      while ((value & 1u) == 0u) {
        position++;
        value >>= 1;
      }
      output_position[i] = position;
    }
  }
}

int main(void) {
  uint32_t input_data[9] = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u, 0u};
  uint32_t output_position[9] = {0u};
  find_first_set(input_data, output_position, 9u);
  return (output_position[0] == 1u && output_position[1] == 2u &&
          output_position[2] == 1u && output_position[3] == 3u &&
          output_position[4] == 1u && output_position[5] == 2u &&
          output_position[6] == 1u && output_position[7] == 4u &&
          output_position[8] == 0u)
             ? 0
             : 1;
}
