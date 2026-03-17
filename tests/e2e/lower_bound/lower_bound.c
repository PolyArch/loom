#include <stdint.h>

__attribute__((noinline)) void
lower_bound(const uint32_t *restrict input_sorted,
            const uint32_t *restrict input_targets,
            uint32_t *restrict output_indices, uint32_t n, uint32_t m) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t t = 0; t < m; ++t) {
    uint32_t target = input_targets[t];
    uint32_t left = 0u;
    uint32_t right = n;

#pragma clang loop vectorize(disable) interleave(disable)
    while (left < right) {
      uint32_t mid = left + (right - left) / 2u;
      if (input_sorted[mid] < target)
        left = mid + 1u;
      else
        right = mid;
    }

    output_indices[t] = left;
  }
}

int main(void) {
  uint32_t input_sorted[5] = {2u, 4u, 6u, 8u, 10u};
  uint32_t input_targets[5] = {1u, 2u, 5u, 8u, 11u};
  uint32_t output_indices[5] = {0u, 0u, 0u, 0u, 0u};
  lower_bound(input_sorted, input_targets, output_indices, 5u, 5u);
  return (output_indices[0] == 0u && output_indices[1] == 0u &&
          output_indices[2] == 2u && output_indices[3] == 3u &&
          output_indices[4] == 5u)
             ? 0
             : 1;
}
