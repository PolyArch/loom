#include <stdint.h>

__attribute__((noinline)) void
rle_decode(const uint32_t *restrict input_values,
           const uint32_t *restrict input_counts,
           uint32_t *restrict output_data, uint32_t encoded_length) {
  uint32_t write_idx = 0u;

#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0u; i < encoded_length; ++i) {
    uint32_t value = input_values[i];
    uint32_t count = input_counts[i];

#pragma clang loop vectorize(disable) interleave(disable)
    for (uint32_t j = 0u; j < count; ++j) {
      output_data[write_idx] = value;
      write_idx++;
    }
  }
}

int main(void) {
  uint32_t input_values[3] = {5u, 3u, 7u};
  uint32_t input_counts[3] = {3u, 2u, 1u};
  uint32_t output_data[6] = {0u, 0u, 0u, 0u, 0u, 0u};

  rle_decode(input_values, input_counts, output_data, 3u);

  return (output_data[0] == 5u && output_data[1] == 5u &&
          output_data[2] == 5u && output_data[3] == 3u &&
          output_data[4] == 3u && output_data[5] == 7u)
             ? 0
             : 1;
}
