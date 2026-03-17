#include <stdint.h>

__attribute__((noinline)) void
delta_encode(const uint32_t *restrict input_data,
             uint32_t *restrict output_deltas, uint32_t n) {
  if (n == 0u)
    return;

  output_deltas[0] = input_data[0];

#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 1; i < n; ++i)
    output_deltas[i] = input_data[i] - input_data[i - 1];
}

int main(void) {
  uint32_t input_data[5] = {100u, 101u, 103u, 106u, 110u};
  uint32_t output_deltas[5] = {0u, 0u, 0u, 0u, 0u};
  delta_encode(input_data, output_deltas, 5u);
  return (output_deltas[0] == 100u && output_deltas[1] == 1u &&
          output_deltas[2] == 2u && output_deltas[3] == 3u &&
          output_deltas[4] == 4u)
             ? 0
             : 1;
}
