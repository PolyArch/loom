// Loom kernel implementation: parity
#include "parity.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Parity bit computation
// Tests complete compilation chain with nested while loops and XOR reduction (different from popcount which uses addition)
// Test: 7 (0b111) → 1, 6 (0b110) → 0, 14 (0b1110) → 1






// CPU implementation of parity computation
// Returns 1 if the number of set bits is odd, 0 if even
void parity_cpu(const uint32_t* __restrict__ input_data,
                uint32_t* __restrict__ output_parity,
                const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        uint32_t value = input_data[i];
        uint32_t parity = 0;
        
        while (value) {
            parity ^= (value & 1);
            value >>= 1;
        }
        
        output_parity[i] = parity;
    }
}

// Parity: output_parity[i] = XOR of all bits in input_data[i] (1 if odd number of 1s, 0 if even)
// Accelerator implementation of parity computation
LOOM_ACCEL()
void parity_dsa(const uint32_t* __restrict__ input_data,
                uint32_t* __restrict__ output_parity,
                const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t i = 0; i < N; i++) {
        uint32_t value = input_data[i];
        uint32_t parity = 0;
        
        while (value) {
            parity ^= (value & 1);
            value >>= 1;
        }
        
        output_parity[i] = parity;
    }
}




