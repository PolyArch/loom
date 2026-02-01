// Loom kernel implementation: bit_reverse
#include "bit_reverse.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Bit reversal within 32-bit values
// Tests complete compilation chain with bit manipulation
// Test: input=[0x12345678,0xABCDEF00,0x1,0x80000000], N=4 → bit-reversed values






// CPU implementation of bit reversal
// Reverses the bit order within each 32-bit value
// MSB becomes LSB, LSB becomes MSB
void bit_reverse_cpu(const uint32_t* __restrict__ input_data,
                     uint32_t* __restrict__ output_reversed,
                     const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        uint32_t value = input_data[i];
        uint32_t result = 0;
        
        for (uint32_t bit = 0; bit < 32; bit++) {
            result = (result << 1) | (value & 1);
            value >>= 1;
        }
        
        output_reversed[i] = result;
    }
}

// Accelerator implementation of bit reversal
LOOM_ACCEL()
void bit_reverse_dsa(const uint32_t* __restrict__ input_data,
                     uint32_t* __restrict__ output_reversed,
                     const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t i = 0; i < N; i++) {
        uint32_t value = input_data[i];
        uint32_t result = 0;
        
        for (uint32_t bit = 0; bit < 32; bit++) {
            result = (result << 1) | (value & 1);
            value >>= 1;
        }
        
        output_reversed[i] = result;
    }
}



