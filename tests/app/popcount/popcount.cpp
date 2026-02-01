// Loom kernel implementation: popcount
#include "popcount.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Population count (Hamming weight)
// Tests complete compilation chain with nested while loops and bitwise operations
// Test: 7 (0b111) → 3, 6 (0b110) → 2, 15 (0b1111) → 4

// CPU implementation of population count (Hamming weight)
// Counts the number of set bits (1s) in each input value
void popcount_cpu(const uint32_t* __restrict__ input_data,
                  uint32_t* __restrict__ output_count,
                  const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        uint32_t value = input_data[i];
        uint32_t count = 0;

        while (value) {
            count += value & 1;
            value >>= 1;
        }

        output_count[i] = count;
    }
}

// Population count: output_count[i] = number of set bits in input_data[i]
// Accelerator implementation of population count (Hamming weight)
LOOM_ACCEL()
void popcount_dsa(const uint32_t* __restrict__ input_data,
                  uint32_t* __restrict__ output_count,
                  const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t i = 0; i < N; i++) {
        uint32_t value = input_data[i];
        uint32_t count = 0;

        while (value) {
            count += value & 1;
            value >>= 1;
        }

        output_count[i] = count;
    }
}

