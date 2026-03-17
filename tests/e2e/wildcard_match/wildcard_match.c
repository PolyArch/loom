#include <stdint.h>

__attribute__((noinline)) void wildcard_match(
    const uint32_t *restrict input_text, const uint32_t *restrict input_pattern,
    uint32_t *restrict output_match, uint32_t n, uint32_t m) {
  const uint32_t wildcard = 63u;
  uint32_t found = 0u;
  uint32_t limit = 0u;

  if (m <= n)
    limit = n - m + 1u;

#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0u; i < limit; ++i) {
    if (found == 0u) {
      uint32_t match = 1u;

#pragma clang loop vectorize(disable) interleave(disable)
      for (uint32_t j = 0u; j < m; ++j) {
        uint32_t pat = input_pattern[j];
        uint32_t text = input_text[i + j];
        if (match != 0u && pat != wildcard && text != pat)
          match = 0u;
      }

      if (match != 0u)
        found = 1u;
    }
  }

  output_match[0] = found;
}

int main(void) {
  uint32_t input_text[6] = {97u, 98u, 99u, 100u, 101u, 102u};
  uint32_t input_pattern[3] = {99u, 63u, 101u};
  uint32_t output_match[1] = {0u};

  wildcard_match(input_text, input_pattern, output_match, 6u, 3u);
  return output_match[0] == 1u ? 0 : 1;
}
