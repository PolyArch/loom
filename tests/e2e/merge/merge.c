#include <stdint.h>

__attribute__((noinline)) void
merge(const uint32_t *restrict input_a, const uint32_t *restrict input_b,
      uint32_t *restrict output, uint32_t n, uint32_t m) {
  uint32_t i = 0u;
  uint32_t j = 0u;
  uint32_t k = 0u;

#pragma clang loop vectorize(disable) interleave(disable)
  while (i < n && j < m) {
    uint32_t a = input_a[i];
    uint32_t b = input_b[j];
    if (a <= b) {
      output[k] = a;
      i++;
    } else {
      output[k] = b;
      j++;
    }
    k++;
  }

#pragma clang loop vectorize(disable) interleave(disable)
  while (i < n) {
    output[k] = input_a[i];
    i++;
    k++;
  }

#pragma clang loop vectorize(disable) interleave(disable)
  while (j < m) {
    output[k] = input_b[j];
    j++;
    k++;
  }
}

int main(void) {
  uint32_t input_a[3] = {1u, 3u, 5u};
  uint32_t input_b[4] = {2u, 4u, 6u, 7u};
  uint32_t output[7] = {0u, 0u, 0u, 0u, 0u, 0u, 0u};
  merge(input_a, input_b, output, 3u, 4u);
  return (output[0] == 1u && output[1] == 2u && output[2] == 3u &&
          output[3] == 4u && output[4] == 5u && output[5] == 6u &&
          output[6] == 7u)
             ? 0
             : 1;
}
