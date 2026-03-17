#include <stdint.h>

__attribute__((noinline)) void
unpack_bits(const uint32_t *restrict input_packed,
            uint32_t *restrict output_bits, uint32_t num_bits) {
  const uint32_t bits_per_word = 32u;
  uint32_t num_words = (num_bits + bits_per_word - 1u) / bits_per_word;

#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t word_idx = 0; word_idx < num_words; ++word_idx) {
    uint32_t packed_word = input_packed[word_idx];
    uint32_t start_bit = word_idx * bits_per_word;
    uint32_t end_bit = start_bit + bits_per_word;
    if (end_bit > num_bits)
      end_bit = num_bits;

#pragma clang loop vectorize(disable) interleave(disable)
    for (uint32_t bit_idx = start_bit; bit_idx < end_bit; ++bit_idx) {
      uint32_t bit_position = bit_idx - start_bit;
      output_bits[bit_idx] = (packed_word >> bit_position) & 1u;
    }
  }
}

int main(void) {
  uint32_t input_packed[2];
  uint32_t output_bits[36];
  input_packed[0] = 1431655765u;
  input_packed[1] = 13u;

#pragma clang loop vectorize(disable) interleave(disable)
  for (uint32_t i = 0; i < 36u; ++i)
    output_bits[i] = 0u;

  unpack_bits(input_packed, output_bits, 36u);
  return (output_bits[0] == 1u && output_bits[1] == 0u &&
          output_bits[2] == 1u && output_bits[3] == 0u &&
          output_bits[32] == 1u && output_bits[33] == 0u &&
          output_bits[34] == 1u && output_bits[35] == 1u)
             ? 0
             : 1;
}
