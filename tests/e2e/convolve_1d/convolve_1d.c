#include <stdint.h>

__attribute__((noinline)) void
convolve_1d(const uint32_t *restrict input_data,
            const uint32_t *restrict kernel_data,
            uint32_t *restrict output_data, uint32_t input_size,
            uint32_t kernel_size) {
  uint32_t output_size = input_size - kernel_size + 1u;

#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t n = 0; n < output_size; ++n) {
    uint32_t sum = 0u;
#pragma clang loop vectorize(disable) interleave(disable)
    for (uint32_t k = 0; k < kernel_size; ++k)
      sum += input_data[n + k] * kernel_data[k];
    output_data[n] = sum;
  }
}

int main(void) {
  uint32_t input_data[6] = {1u, 2u, 3u, 4u, 5u, 6u};
  uint32_t kernel_data[3] = {1u, 2u, 3u};
  uint32_t output_data[4] = {0u, 0u, 0u, 0u};
  convolve_1d(input_data, kernel_data, output_data, 6u, 3u);
  return (output_data[0] == 14u && output_data[1] == 20u &&
          output_data[2] == 26u && output_data[3] == 32u)
             ? 0
             : 1;
}
