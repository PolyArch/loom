// Loom kernel implementation: unpack_bits
#include "unpack_bits.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>







// CPU implementation of bit unpacking
void unpack_bits_cpu(const uint32_t* __restrict__ input_packed,
                     uint32_t* __restrict__ output_bits,
                     const uint32_t num_bits) {
    const uint32_t bits_per_word = 32;
    uint32_t num_words = (num_bits + bits_per_word - 1) / bits_per_word;
    
    for (uint32_t word_idx = 0; word_idx < num_words; word_idx++) {
        uint32_t packed_word = input_packed[word_idx];
        uint32_t start_bit = word_idx * bits_per_word;
        uint32_t end_bit = start_bit + bits_per_word;
        if (end_bit > num_bits) {
            end_bit = num_bits;
        }
        
        for (uint32_t bit_idx = start_bit; bit_idx < end_bit; bit_idx++) {
            uint32_t bit_position = bit_idx - start_bit;
            output_bits[bit_idx] = (packed_word >> bit_position) & 1;
        }
    }
}

// Accelerator implementation of bit unpacking
LOOM_ACCEL()
void unpack_bits_dsa(LOOM_MEMORY_BANK(8) LOOM_STREAM const uint32_t* __restrict__ input_packed,
                     LOOM_STREAM uint32_t* __restrict__ output_bits,
                     const uint32_t num_bits) {
    const uint32_t bits_per_word = 32;
    uint32_t num_words = (num_bits + bits_per_word - 1) / bits_per_word;
    
    for (uint32_t word_idx = 0; word_idx < num_words; word_idx++) {
        uint32_t packed_word = input_packed[word_idx];
        uint32_t start_bit = word_idx * bits_per_word;
        uint32_t end_bit = start_bit + bits_per_word;
        if (end_bit > num_bits) {
            end_bit = num_bits;
        }
        
        for (uint32_t bit_idx = start_bit; bit_idx < end_bit; bit_idx++) {
            uint32_t bit_position = bit_idx - start_bit;
            output_bits[bit_idx] = (packed_word >> bit_position) & 1;
        }
    }
}



