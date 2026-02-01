// Loom kernel implementation: bitrev
#include "bitrev.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Bit-reversal permutation
// Tests complete compilation chain with bit-reversed index permutation
// Test: input=[1..8], N=8 → bit-reversed permutation

// CPU implementation of bit-reversal permutation
// Reorder array elements according to bit-reversed indices
void bitrev_cpu(const float* __restrict__ input,
                float* __restrict__ output,
                const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = 0;
        uint32_t k = i;
        uint32_t m = N >> 1;

        // Compute bit-reversed index
        while (m > 0) {
            j = (j << 1) | (k & 1);
            k >>= 1;
            m >>= 1;
        }

        output[j] = input[i];
    }
}

// Accelerator implementation of bit-reversal permutation
LOOM_ACCEL()
void bitrev_dsa(const float* __restrict__ input,
                float* __restrict__ output,
                const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = 0;
        uint32_t k = i;
        uint32_t m = N >> 1;

        while (m > 0) {
            j = (j << 1) | (k & 1);
            k >>= 1;
            m >>= 1;
        }

        output[j] = input[i];
    }
}

