#include <stdint.h>

__attribute__((noinline)) uint32_t database_join(
    const int32_t *restrict a_ids, const int32_t *restrict b_ids,
    const int32_t *restrict a_values, const int32_t *restrict b_values,
    int32_t *restrict output_ids, int32_t *restrict output_a_values,
    int32_t *restrict output_b_values, uint32_t size_a, uint32_t size_b) {
  uint32_t out_idx = 0u;

#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0u; i < size_a; ++i) {
#pragma clang loop vectorize(disable) interleave(disable)
    for (uint32_t j = 0u; j < size_b; ++j) {
      if (a_ids[i] == b_ids[j]) {
        output_ids[out_idx] = a_ids[i];
        output_a_values[out_idx] = a_values[i];
        output_b_values[out_idx] = b_values[j];
        out_idx++;
      }
    }
  }

  return out_idx;
}

int main(void) {
  int32_t a_ids[3] = {1, 2, 3};
  int32_t b_ids[3] = {2, 3, 4};
  int32_t a_values[3] = {10, 20, 30};
  int32_t b_values[3] = {200, 300, 400};
  int32_t output_ids[3] = {0, 0, 0};
  int32_t output_a_values[3] = {0, 0, 0};
  int32_t output_b_values[3] = {0, 0, 0};

  uint32_t count =
      database_join(a_ids, b_ids, a_values, b_values, output_ids,
                    output_a_values, output_b_values, 3u, 3u);

  return (count == 2u && output_ids[0] == 2 && output_ids[1] == 3 &&
          output_a_values[0] == 20 && output_a_values[1] == 30 &&
          output_b_values[0] == 200 && output_b_values[1] == 300)
             ? 0
             : 1;
}
