#include <stdint.h>

__attribute__((noinline)) void pack_bits(const uint32_t *restrict input_bits,
                                         uint32_t *restrict output_packed,
                                         uint32_t num_bits) {
  const uint32_t bits_per_word = 32u;
  uint32_t num_words = (num_bits + bits_per_word - 1u) / bits_per_word;

#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t word_idx = 0; word_idx < num_words; ++word_idx) {
    uint32_t packed_word = 0u;
    uint32_t start_bit = word_idx * bits_per_word;
    uint32_t end_bit = start_bit + bits_per_word;
    if (end_bit > num_bits)
      end_bit = num_bits;

#pragma clang loop vectorize(disable) interleave(disable)
    for (uint32_t bit_idx = start_bit; bit_idx < end_bit; ++bit_idx) {
      uint32_t bit_position = bit_idx - start_bit;
      if ((input_bits[bit_idx] & 1u) != 0u)
        packed_word |= (1u << bit_position);
    }

    output_packed[word_idx] = packed_word;
  }
}

int main(void) {
  uint32_t input_bits[36] = {
      1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u,
      1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u,
      1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 0u, 1u, 1u,
  };
  uint32_t output_packed[2] = {0u, 0u};
  pack_bits(input_bits, output_packed, 36u);
  return (output_packed[0] == 0x55555555u && output_packed[1] == 13u) ? 0 : 1;
}
