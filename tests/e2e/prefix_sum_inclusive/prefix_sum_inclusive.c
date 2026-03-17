#include <stdint.h>

__attribute__((noinline)) void
prefix_sum_inclusive(const uint32_t *restrict input,
                     uint32_t *restrict output, uint32_t n) {
  uint32_t sum = 0u;

#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0; i < n; ++i) {
    sum += input[i];
    output[i] = sum;
  }
}

int main(void) {
  uint32_t input[5] = {1u, 2u, 3u, 4u, 5u};
  uint32_t output[5] = {0u, 0u, 0u, 0u, 0u};
  prefix_sum_inclusive(input, output, 5u);
  return (output[0] == 1u && output[1] == 3u && output[2] == 6u &&
          output[3] == 10u && output[4] == 15u)
             ? 0
             : 1;
}
