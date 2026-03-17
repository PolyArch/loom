#include <stdint.h>

__attribute__((noinline)) void
stream_update(const uint32_t *restrict input, uint32_t *restrict output,
              uint32_t n, uint32_t step) {
  uint32_t acc = 0u;

#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0u; i + step < n; i += step) {
#pragma clang loop vectorize(disable) interleave(disable)
    for (uint32_t j = n; j > 0u; j >>= 1u) {
      uint32_t idx = (i + j) % n;
      acc += input[idx];
    }
  }

  output[0] = acc;
}

int main(void) {
  uint32_t input[8] = {3u, 1u, 4u, 1u, 5u, 9u, 2u, 6u};
  uint32_t output[1] = {0u};
  stream_update(input, output, 8u, 2u);
  return output[0] == 61u ? 0 : 1;
}
