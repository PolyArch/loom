#include <stdint.h>

__attribute__((noinline)) void
delta_decode(const uint32_t *restrict input_deltas,
             uint32_t *restrict output_data, uint32_t n) {
  if (n == 0u)
    return;

  output_data[0] = input_deltas[0];

#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 1; i < n; ++i)
    output_data[i] = output_data[i - 1] + input_deltas[i];
}

int main(void) {
  uint32_t input_deltas[5] = {100u, 1u, 2u, 3u, 4u};
  uint32_t output_data[5] = {0u, 0u, 0u, 0u, 0u};
  delta_decode(input_deltas, output_data, 5u);
  return (output_data[0] == 100u && output_data[1] == 101u &&
          output_data[2] == 103u && output_data[3] == 106u &&
          output_data[4] == 110u)
             ? 0
             : 1;
}
