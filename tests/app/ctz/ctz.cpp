// Loom kernel implementation: ctz
#include "ctz.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Count trailing zeros
// Tests complete compilation chain with conditionals, while loops, and bitwise operations
// Test: input=[0x8,0x10,0x100,0x10000,0x1000000,0], N=6 → output=[3,4,8,16,24,32]

// CPU implementation of count trailing zeros
// Returns the number of trailing zero bits (from LSB)
// For input value 0, returns 32
void ctz_cpu(const uint32_t* __restrict__ input_data,
             uint32_t* __restrict__ output_count,
             const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        uint32_t value = input_data[i];

        if (value == 0) {
            output_count[i] = 32;
        } else {
            uint32_t count = 0;

            while ((value & 1) == 0) {
                count++;
                value >>= 1;
            }

            output_count[i] = count;
        }
    }
}

// Count trailing zeros: output_count[i] = number of trailing zero bits in input_data[i]
// Accelerator implementation of count trailing zeros
LOOM_ACCEL()
void ctz_dsa(const uint32_t* __restrict__ input_data,
             uint32_t* __restrict__ output_count,
             const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t i = 0; i < N; i++) {
        uint32_t value = input_data[i];

        if (value == 0) {
            output_count[i] = 32;
        } else {
            uint32_t count = 0;

            while ((value & 1) == 0) {
                count++;
                value >>= 1;
            }

            output_count[i] = count;
        }
    }
}

