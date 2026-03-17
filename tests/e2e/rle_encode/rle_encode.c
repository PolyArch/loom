#include <stdint.h>

__attribute__((noinline)) uint32_t
rle_encode(const uint32_t *restrict input_data,
           uint32_t *restrict output_values,
           uint32_t *restrict output_counts, uint32_t n) {
  if (n == 0u)
    return 0u;

  uint32_t write_idx = 0u;
  uint32_t current_value = input_data[0];
  uint32_t current_count = 1u;

#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 1u; i < n; ++i) {
    uint32_t value = input_data[i];
    if (value == current_value) {
      current_count++;
    } else {
      output_values[write_idx] = current_value;
      output_counts[write_idx] = current_count;
      write_idx++;
      current_value = value;
      current_count = 1u;
    }
  }

  output_values[write_idx] = current_value;
  output_counts[write_idx] = current_count;
  write_idx++;
  return write_idx;
}

int main(void) {
  uint32_t input_data[6] = {5u, 5u, 5u, 3u, 3u, 7u};
  uint32_t output_values[6] = {0u, 0u, 0u, 0u, 0u, 0u};
  uint32_t output_counts[6] = {0u, 0u, 0u, 0u, 0u, 0u};
  uint32_t length = rle_encode(input_data, output_values, output_counts, 6u);
  return (length == 3u && output_values[0] == 5u && output_values[1] == 3u &&
          output_values[2] == 7u && output_counts[0] == 3u &&
          output_counts[1] == 2u && output_counts[2] == 1u)
             ? 0
             : 1;
}
