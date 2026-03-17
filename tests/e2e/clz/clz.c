#include <stdint.h>

__attribute__((noinline)) void clz(const uint32_t *restrict input_data,
                                   uint32_t *restrict output_count,
                                   uint32_t n) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0; i < n; ++i) {
    uint32_t value = input_data[i];

    if (value == 0u) {
      output_count[i] = 32u;
    } else {
      uint32_t count = 0u;
      uint32_t mask = 0x80000000u;
      while ((value & mask) == 0u) {
        count++;
        mask >>= 1;
      }
      output_count[i] = count;
    }
  }
}

int main(void) {
  uint32_t input_data[6] = {1u, 0x100u, 0x10000u, 0x1000000u, 0x80000000u, 0u};
  uint32_t output_count[6] = {0u, 0u, 0u, 0u, 0u, 0u};
  clz(input_data, output_count, 6u);
  return (output_count[0] == 31u && output_count[1] == 23u &&
          output_count[2] == 15u && output_count[3] == 7u &&
          output_count[4] == 0u && output_count[5] == 32u)
             ? 0
             : 1;
}
