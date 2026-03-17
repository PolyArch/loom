#include <stdint.h>

__attribute__((noinline)) void
kmp_table(const uint32_t *restrict input_pattern,
          uint32_t *restrict output_table, uint32_t m) {
  output_table[0] = 0u;
  uint32_t j = 0u;

#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 1u; i < m; ++i) {
    uint32_t search = 1u;

#pragma clang loop vectorize(disable) interleave(disable)
    while (search != 0u) {
      if (j != 0u && input_pattern[i] != input_pattern[j]) {
        uint32_t prev = j - 1u;
        j = output_table[prev];
      } else {
        search = 0u;
      }
    }

    if (input_pattern[i] == input_pattern[j])
      j++;

    output_table[i] = j;
  }
}

int main(void) {
  uint32_t input_pattern[6] = {65u, 66u, 65u, 66u, 65u, 67u};
  uint32_t output_table[6] = {0u, 0u, 0u, 0u, 0u, 0u};
  kmp_table(input_pattern, output_table, 6u);
  return (output_table[0] == 0u && output_table[1] == 0u &&
          output_table[2] == 1u && output_table[3] == 2u &&
          output_table[4] == 3u && output_table[5] == 0u)
             ? 0
             : 1;
}
