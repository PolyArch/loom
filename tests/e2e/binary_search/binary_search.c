#include <stdint.h>

__attribute__((noinline)) void
binary_search(const uint32_t *restrict input_sorted,
              const uint32_t *restrict input_targets,
              uint32_t *restrict output_indices, uint32_t n, uint32_t m) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t t = 0; t < m; ++t) {
    uint32_t target = input_targets[t];
    uint32_t left = 0u;
    uint32_t right = n;
    uint32_t found = 0xffffffffu;

#pragma clang loop vectorize(disable) interleave(disable)
    while (left < right) {
      uint32_t mid = left + (right - left) / 2u;
      uint32_t value = input_sorted[mid];
      if (value == target) {
        found = mid;
        left = right;
      } else if (value < target) {
        left = mid + 1u;
      } else {
        right = mid;
      }
    }

    output_indices[t] = found;
  }
}

int main(void) {
  uint32_t input_sorted[8] = {1u, 3u, 5u, 7u, 9u, 11u, 13u, 15u};
  uint32_t input_targets[5] = {5u, 13u, 2u, 15u, 20u};
  uint32_t output_indices[5] = {0u, 0u, 0u, 0u, 0u};
  binary_search(input_sorted, input_targets, output_indices, 8u, 5u);
  return (output_indices[0] == 2u && output_indices[1] == 6u &&
          output_indices[2] == 0xffffffffu && output_indices[3] == 7u &&
          output_indices[4] == 0xffffffffu)
             ? 0
             : 1;
}
