#include <stdint.h>

void popcount(const uint32_t *restrict input_data,
              uint32_t *restrict output_count, uint32_t n) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t value = input_data[i];
    uint32_t count = 0;

    while (value != 0u) {
      count += value & 1u;
      value >>= 1;
    }

    output_count[i] = count;
  }
}

int main(void) {
  uint32_t input_data[4] = {7u, 6u, 15u, 16u};
  uint32_t output_count[4] = {0u, 0u, 0u, 0u};
  popcount(input_data, output_count, 4u);
  return (output_count[0] == 3u && output_count[1] == 2u &&
          output_count[2] == 4u && output_count[3] == 1u)
             ? 0
             : 1;
}
