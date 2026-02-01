// Loom kernel implementation: pack_bits
#include "pack_bits.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Bit packing into 32-bit words
// Tests complete compilation chain with bit manipulation and nested loops
// Test: [1,0,1,1,0,0,0,0] → packed word = 0b00001101 = 13

// CPU implementation of bit packing
void pack_bits_cpu(const uint32_t* __restrict__ input_bits,
                   uint32_t* __restrict__ output_packed,
                   const uint32_t num_bits) {
    const uint32_t bits_per_word = 32;
    uint32_t num_words = (num_bits + bits_per_word - 1) / bits_per_word;

    for (uint32_t word_idx = 0; word_idx < num_words; word_idx++) {
        uint32_t packed_word = 0;
        uint32_t start_bit = word_idx * bits_per_word;
        uint32_t end_bit = start_bit + bits_per_word;
        if (end_bit > num_bits) {
            end_bit = num_bits;
        }

        for (uint32_t bit_idx = start_bit; bit_idx < end_bit; bit_idx++) {
            uint32_t bit_position = bit_idx - start_bit;
            if (input_bits[bit_idx] & 1) {
                packed_word |= (1u << bit_position);
            }
        }

        output_packed[word_idx] = packed_word;
    }
}

// Accelerator implementation of bit packing
LOOM_ACCEL()
void pack_bits_dsa(LOOM_MEMORY_BANK(8) LOOM_STREAM const uint32_t* __restrict__ input_bits,
                   LOOM_STREAM uint32_t* __restrict__ output_packed,
                   const uint32_t num_bits) {
    const uint32_t bits_per_word = 32;
    uint32_t num_words = (num_bits + bits_per_word - 1) / bits_per_word;

    for (uint32_t word_idx = 0; word_idx < num_words; word_idx++) {
        uint32_t packed_word = 0;
        uint32_t start_bit = word_idx * bits_per_word;
        uint32_t end_bit = start_bit + bits_per_word;
        if (end_bit > num_bits) {
            end_bit = num_bits;
        }

        for (uint32_t bit_idx = start_bit; bit_idx < end_bit; bit_idx++) {
            uint32_t bit_position = bit_idx - start_bit;
            if (input_bits[bit_idx] & 1) {
                packed_word |= (1u << bit_position);
            }
        }

        output_packed[word_idx] = packed_word;
    }
}

