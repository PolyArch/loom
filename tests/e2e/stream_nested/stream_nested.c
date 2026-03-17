#include <stdint.h>

__attribute__((noinline)) void
stream_nested(const uint32_t *restrict input, uint32_t *restrict output,
              uint32_t n) {
  uint32_t acc = 0u;

#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 1u; i < n; i <<= 1u) {
#pragma clang loop vectorize(disable) interleave(disable)
    for (uint32_t j = n; j != 0u; j >>= 1u) {
#pragma clang loop vectorize(disable) interleave(disable)
      for (uint32_t k = 1u; k <= n; k <<= 1u) {
        uint32_t idx = (i + j + k) % n;
        acc += input[idx];
      }
    }
  }

  output[0] = acc;
}

int main(void) {
  uint32_t input[8] = {3u, 1u, 4u, 1u, 5u, 9u, 2u, 6u};
  uint32_t output[1] = {0u};
  stream_nested(input, output, 8u);
  return output[0] == 198u ? 0 : 1;
}
