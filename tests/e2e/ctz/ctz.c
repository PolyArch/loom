#include <stdint.h>

__attribute__((noinline)) void ctz(const uint32_t *restrict input_data,
                                   uint32_t *restrict output_count,
                                   uint32_t n) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t value = input_data[i];

    if (value == 0u) {
      output_count[i] = 32u;
    } else {
      uint32_t count = 0u;
      while ((value & 1u) == 0u) {
        count++;
        value >>= 1;
      }
      output_count[i] = count;
    }
  }
}

int main(void) {
  uint32_t input_data[6] = {0x8u, 0x10u, 0x100u, 0x10000u, 0x1000000u, 0u};
  uint32_t output_count[6] = {0u, 0u, 0u, 0u, 0u, 0u};
  ctz(input_data, output_count, 6u);
  return (output_count[0] == 3u && output_count[1] == 4u &&
          output_count[2] == 8u && output_count[3] == 16u &&
          output_count[4] == 24u && output_count[5] == 32u)
             ? 0
             : 1;
}
