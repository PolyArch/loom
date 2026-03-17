#include <stdint.h>

__attribute__((noinline)) uint32_t
compact_predicate(const uint32_t *restrict input,
                  const uint32_t *restrict predicate,
                  uint32_t *restrict output, uint32_t n) {
  uint32_t count = 0u;

#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0u; i < n; ++i) {
    if (predicate[i] != 0u) {
      output[count] = input[i];
      count++;
    }
  }

  return count;
}

int main(void) {
  uint32_t input[8] = {10u, 20u, 30u, 40u, 50u, 60u, 70u, 80u};
  uint32_t predicate[8] = {1u, 0u, 1u, 0u, 1u, 1u, 0u, 1u};
  uint32_t output[8] = {0u, 0u, 0u, 0u, 0u, 0u, 0u, 0u};
  uint32_t count = compact_predicate(input, predicate, output, 8u);
  return (count == 5u && output[0] == 10u && output[1] == 30u &&
          output[2] == 50u && output[3] == 60u && output[4] == 80u)
             ? 0
             : 1;
}
