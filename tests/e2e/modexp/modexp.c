#include <stdint.h>

__attribute__((noinline)) void
modexp(const uint32_t *restrict input_base, const uint32_t *restrict input_exp,
       uint32_t *restrict output_result, uint32_t modulus, uint32_t n) {
#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0u; i < n; ++i) {
    uint32_t result = 1u;
    uint32_t base = input_base[i] % modulus;
    uint32_t exp = input_exp[i];

#pragma clang loop vectorize(disable) interleave(disable)
    while (exp > 0u) {
      if ((exp & 1u) != 0u)
        result = (result * base) % modulus;
      base = (base * base) % modulus;
      exp >>= 1u;
    }

    output_result[i] = result;
  }
}

int main(void) {
  uint32_t input_base[4] = {2u, 3u, 5u, 7u};
  uint32_t input_exp[4] = {3u, 4u, 2u, 5u};
  uint32_t output_result[4] = {0u, 0u, 0u, 0u};
  modexp(input_base, input_exp, output_result, 7u, 4u);
  return (output_result[0] == 1u && output_result[1] == 4u &&
          output_result[2] == 4u && output_result[3] == 0u)
             ? 0
             : 1;
}
