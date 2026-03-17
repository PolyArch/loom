#include <stdint.h>

__attribute__((noinline)) uint32_t
vecnorm_l2(const uint32_t *restrict input_data, uint32_t n) {
  uint32_t norm_sq = 0u;

#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0; i < n; ++i)
    norm_sq += input_data[i] * input_data[i];

  return norm_sq;
}

int main(void) {
  uint32_t input_data[8] = {1u, 2u, 3u, 4u, 5u, 6u, 7u, 8u};
  return vecnorm_l2(input_data, 8u) == 204u ? 0 : 1;
}
