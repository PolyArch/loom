#include <stdint.h>

__attribute__((noinline)) uint32_t
string_compare(const uint32_t *restrict input_str_a,
               const uint32_t *restrict input_str_b, uint32_t n) {
  uint32_t result = 0u;
  uint32_t done = 0u;

#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0u; i < n; ++i) {
    if (done == 0u) {
      if (input_str_a[i] < input_str_b[i]) {
        result = 0xFFFFFFFFu;
        done = 1u;
      } else if (input_str_a[i] > input_str_b[i]) {
        result = 1u;
        done = 1u;
      }
    }
  }
  return result;
}

int main(void) {
  uint32_t input_str_a[8] = {97u, 112u, 112u, 108u, 101u, 116u, 101u, 97u};
  uint32_t input_str_b[8] = {97u, 112u, 112u, 108u, 101u, 116u, 111u, 110u};
  uint32_t result = string_compare(input_str_a, input_str_b, 8u);
  return result == 0xFFFFFFFFu ? 0 : 1;
}
