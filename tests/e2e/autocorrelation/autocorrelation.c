#include <stdint.h>

__attribute__((noinline)) void
autocorrelation(const uint32_t *restrict input_data,
                uint32_t *restrict output_data, uint32_t x_size,
                uint32_t max_lag) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t lag = 0; lag < max_lag; ++lag) {
    uint32_t sum = 0u;
#pragma clang loop vectorize(disable) interleave(disable)
    for (uint32_t i = 0; i < x_size - lag; ++i)
      sum += input_data[i] * input_data[i + lag];
    output_data[lag] = sum;
  }
}

int main(void) {
  uint32_t input_data[5] = {1u, 2u, 3u, 4u, 5u};
  uint32_t output_data[3] = {0u, 0u, 0u};
  autocorrelation(input_data, output_data, 5u, 3u);
  return (output_data[0] == 55u && output_data[1] == 40u &&
          output_data[2] == 26u)
             ? 0
             : 1;
}
