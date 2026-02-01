// Loom kernel implementation: clz
#include "clz.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Count leading zeros
// Tests complete compilation chain with conditionals, while loops, and bitwise operations
// Test: input=[0x1,0x100,0x10000,0x1000000,0x80000000,0], N=6 → output=[31,23,15,7,0,32]

// CPU implementation of count leading zeros
// Returns the number of leading zero bits (from MSB)
// For input value 0, returns 32
void clz_cpu(const uint32_t* __restrict__ input_data,
             uint32_t* __restrict__ output_count,
             const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        uint32_t value = input_data[i];

        if (value == 0) {
            output_count[i] = 32;
        } else {
            uint32_t count = 0;
            uint32_t mask = 0x80000000;

            while ((value & mask) == 0) {
                count++;
                mask >>= 1;
            }

            output_count[i] = count;
        }
    }
}

// Count leading zeros: output_count[i] = number of leading zero bits in input_data[i]
// Accelerator implementation of count leading zeros
LOOM_ACCEL()
void clz_dsa(const uint32_t* __restrict__ input_data,
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
            uint32_t mask = 0x80000000;

            while ((value & mask) == 0) {
                count++;
                mask >>= 1;
            }

            output_count[i] = count;
        }
    }
}

