#include <stdint.h>

__attribute__((noinline)) uint32_t
partition(const uint32_t *restrict input, uint32_t *restrict output, uint32_t n,
          uint32_t pivot) {
  uint32_t write_pos = 0u;

#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0u; i < n; ++i) {
    uint32_t value = input[i];
    if (value <= pivot) {
      output[write_pos] = value;
      write_pos++;
    }
  }

#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0u; i < n; ++i) {
    uint32_t value = input[i];
    if (value > pivot) {
      output[write_pos] = value;
      write_pos++;
    }
  }

  return write_pos;
}

int main(void) {
  uint32_t input[6] = {5u, 2u, 8u, 3u, 9u, 1u};
  uint32_t output[6] = {0u, 0u, 0u, 0u, 0u, 0u};
  uint32_t count = partition(input, output, 6u, 5u);
  return (count == 6u && output[0] == 5u && output[1] == 2u &&
          output[2] == 3u && output[3] == 1u && output[4] == 8u &&
          output[5] == 9u)
             ? 0
             : 1;
}
