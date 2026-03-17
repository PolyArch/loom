#include <stdint.h>

__attribute__((noinline)) void
string_hash(const uint32_t *restrict input_str,
            uint32_t *restrict output_hashes, uint32_t n,
            uint32_t window_size) {
  const uint32_t base = 256u;
  const uint32_t modulus = 101u;
  if (window_size > n)
    return;

  uint32_t h = 1u;
#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0u; i + 1u < window_size; ++i)
    h = (h * base) % modulus;

  uint32_t hash_value = 0u;
#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0u; i < window_size; ++i)
    hash_value = (hash_value * base + input_str[i]) % modulus;
  output_hashes[0] = hash_value;

#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 1u; i + window_size <= n; ++i) {
    hash_value =
        (hash_value + modulus - ((input_str[i - 1u] * h) % modulus)) % modulus;
    hash_value = (hash_value * base + input_str[i + window_size - 1u]) % modulus;
    output_hashes[i] = hash_value;
  }
}

int main(void) {
  uint32_t input_str[5] = {1u, 2u, 3u, 4u, 5u};
  uint32_t output_hashes[5] = {0u, 0u, 0u, 0u, 0u};
  string_hash(input_str, output_hashes, 5u, 3u);
  return (output_hashes[0] == 98u && output_hashes[1] == 39u &&
          output_hashes[2] == 81u)
             ? 0
             : 1;
}
