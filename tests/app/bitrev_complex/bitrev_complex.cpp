// Loom kernel implementation: bitrev_complex
#include "bitrev_complex.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Bit-reversal for complex numbers
// Tests complete compilation chain with bit reversal permutation on two arrays
// Test: real=[1..8], imag=[0.1..0.8], N=8 → bit-reversed permutation

// CPU implementation of bit-reversal for complex numbers (two arrays)
void bitrev_complex_cpu(const float* __restrict__ input_real,
                        const float* __restrict__ input_imag,
                        float* __restrict__ output_real,
                        float* __restrict__ output_imag,
                        const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        uint32_t j = 0;
        uint32_t k = i;
        uint32_t m = N >> 1;

        while (m > 0) {
            j = (j << 1) | (k & 1);
            k >>= 1;
            m >>= 1;
        }

        output_real[j] = input_real[i];
        output_imag[j] = input_imag[i];
    }
}

// Accelerator implementation of bit-reversal for complex numbers
LOOM_ACCEL()
void bitrev_complex_dsa(const float* __restrict__ input_real,
                        const float* __restrict__ input_imag,
                        float* __restrict__ output_real,
                        float* __restrict__ output_imag,
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

        output_real[j] = input_real[i];
        output_imag[j] = input_imag[i];
    }
}

