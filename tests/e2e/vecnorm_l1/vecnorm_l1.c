#include <stdint.h>

__attribute__((noinline)) uint32_t
vecnorm_l1(const uint32_t *restrict input_data, uint32_t n) {
  uint32_t norm = 0u;

#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0; i < n; ++i)
    norm += input_data[i];

  return norm;
}

int main(void) {
  uint32_t input_data[10] = {15u, 7u, 22u, 9u, 18u,
                             4u,  31u, 12u, 26u, 13u};
  return vecnorm_l1(input_data, 10u) == 157u ? 0 : 1;
}
