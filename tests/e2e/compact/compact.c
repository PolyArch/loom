#include <stdint.h>

__attribute__((noinline)) uint32_t
compact(const uint32_t *restrict input, uint32_t *restrict output, uint32_t n) {
  uint32_t count = 0u;

#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0u; i < n; ++i) {
    uint32_t value = input[i];
    if (value != 0u) {
      output[count] = value;
      count++;
    }
  }

  return count;
}

int main(void) {
  uint32_t input[8] = {10u, 0u, 20u, 0u, 30u, 40u, 0u, 50u};
  uint32_t output[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
  uint32_t count = compact(input, output, 8u);
  return (count == 5u && output[0] == 10u && output[1] == 20u &&
          output[2] == 30u && output[3] == 40u && output[4] == 50u)
             ? 0
             : 1;
}
