// Loom kernel implementation: rotate_bits
#include "rotate_bits.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Bit rotation (left rotation)
// Tests complete compilation chain with bitwise shift operations
// Test: 0xF0F0F0F0<<4, 8<<1, 0xAAAAAAAA<<8






// CPU implementation of bit rotation (left rotation)
// Rotates each 32-bit input left by the corresponding rotation amount
void rotate_bits_cpu(const uint32_t* __restrict__ input_data,
                     const uint32_t* __restrict__ input_shift,
                     uint32_t* __restrict__ output_result,
                     const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        uint32_t value = input_data[i];
        uint32_t shift = input_shift[i] & 0x1F; // Shift amount mod 32
        // Handle rotation correctly for all shift values including 0
        output_result[i] = (shift == 0) ? value : ((value << shift) | (value >> (32 - shift)));
    }
}

// Accelerator implementation of bit rotation (left rotation)
LOOM_ACCEL()
void rotate_bits_dsa(const uint32_t* __restrict__ input_data,
                     const uint32_t* __restrict__ input_shift,
                     uint32_t* __restrict__ output_result,
                     const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t i = 0; i < N; i++) {
        uint32_t value = input_data[i];
        uint32_t shift = input_shift[i] & 0x1F; // Shift amount mod 32
        // Handle rotation correctly for all shift values including 0
        output_result[i] = (shift == 0) ? value : ((value << shift) | (value >> (32 - shift)));
    }
}



