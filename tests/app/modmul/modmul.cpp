// Loom kernel implementation: modmul
#include "modmul.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Modular multiplication (a*b mod n)
// Tests complete compilation chain with modular arithmetic
// Test: 5*3 mod 11 = 4, 7*4 mod 11 = 6, 10*8 mod 11 = 3

// CPU implementation of modular multiplication (a*b mod n)
void modmul_cpu(const uint32_t* __restrict__ input_A,
                const uint32_t* __restrict__ input_B,
                uint32_t* __restrict__ output_C,
                const uint32_t modulus,
                const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        uint64_t a = input_A[i];
        uint64_t b = input_B[i];
        uint64_t result = (a * b) % modulus;
        output_C[i] = (uint32_t)result;
    }
}

// Accelerator implementation of modular multiplication (a*b mod n)
LOOM_ACCEL()
void modmul_dsa(const uint32_t* __restrict__ input_A,
                const uint32_t* __restrict__ input_B,
                uint32_t* __restrict__ output_C,
                const uint32_t modulus,
                const uint32_t N) {
    LOOM_NO_PARALLEL
    LOOM_NO_UNROLL
    for (uint32_t i = 0; i < N; i++) {
        uint64_t a = input_A[i];
        uint64_t b = input_B[i];
        uint64_t result = (a * b) % modulus;
        output_C[i] = (uint32_t)result;
    }
}

