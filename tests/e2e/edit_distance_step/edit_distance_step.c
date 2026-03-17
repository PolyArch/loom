#include <stdint.h>

__attribute__((noinline)) void edit_distance_step(
    const uint32_t *restrict input_left, const uint32_t *restrict input_top,
    const uint32_t *restrict input_diag, const uint32_t *restrict input_char_a,
    const uint32_t *restrict input_char_b, uint32_t *restrict output_result,
    uint32_t n) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0u; i < n; ++i) {
    uint32_t cost = input_char_a[i] == input_char_b[i] ? 0u : 1u;
    uint32_t insert_cost = input_top[i] + 1u;
    uint32_t delete_cost = input_left[i] + 1u;
    uint32_t subst_cost = input_diag[i] + cost;
    uint32_t min_val = insert_cost;

    if (delete_cost < min_val)
      min_val = delete_cost;
    if (subst_cost < min_val)
      min_val = subst_cost;

    output_result[i] = min_val;
  }
}

int main(void) {
  uint32_t input_left[4] = {1u, 2u, 3u, 7u};
  uint32_t input_top[4] = {1u, 2u, 3u, 2u};
  uint32_t input_diag[4] = {0u, 1u, 2u, 6u};
  uint32_t input_char_a[4] = {7u, 8u, 9u, 10u};
  uint32_t input_char_b[4] = {7u, 6u, 5u, 10u};
  uint32_t output_result[4] = {0u, 0u, 0u, 0u};

  edit_distance_step(input_left, input_top, input_diag, input_char_a,
                     input_char_b, output_result, 4u);

  return (output_result[0] == 0u && output_result[1] == 2u &&
          output_result[2] == 3u && output_result[3] == 2u)
             ? 0
             : 1;
}
