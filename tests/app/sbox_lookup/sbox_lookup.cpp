// Loom kernel implementation: sbox_lookup
#include "sbox_lookup.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: S-box substitution table lookup
// Tests complete compilation chain with indirect memory access
// Test: input=[0,1,2,255], sbox[0]=100, sbox[1]=200, sbox[2]=150, sbox[255]=255






// CPU implementation of S-box substitution table lookup
// Takes input values, an S-box table, and performs table lookup
void sbox_lookup_cpu(const uint32_t* __restrict__ input_data,
                     const uint32_t* __restrict__ input_sbox,
                     uint32_t* __restrict__ output_result,
                     const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        uint32_t index = input_data[i] & 0xFF; // Use lower 8 bits as index
        output_result[i] = input_sbox[index];
    }
}

// Accelerator implementation of S-box substitution table lookup
LOOM_ACCEL()
void sbox_lookup_dsa(const uint32_t* __restrict__ input_data,
                     const uint32_t* __restrict__ input_sbox,
                     uint32_t* __restrict__ output_result,
                     const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t i = 0; i < N; i++) {
        uint32_t index = input_data[i] & 0xFF; // Use lower 8 bits as index
        output_result[i] = input_sbox[index];
    }
}



