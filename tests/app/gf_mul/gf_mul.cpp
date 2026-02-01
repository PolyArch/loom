// Loom kernel implementation: gf_mul
#include "gf_mul.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Galois Field GF(2^8) multiplication
// Tests complete compilation chain with bitwise operations (shift, XOR, AND)
// Test: A=[83,202,1] * B=[202,83,5] in GF(2^8) → [1,1,5]






// CPU implementation of Galois Field GF(2^8) multiplication
// Uses AES irreducible polynomial: x^8 + x^4 + x^3 + x + 1 (0x11B)
void gf_mul_cpu(const uint32_t* __restrict__ input_A,
                const uint32_t* __restrict__ input_B,
                uint32_t* __restrict__ output_C,
                const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        uint32_t a = input_A[i] & 0xFF;
        uint32_t b = input_B[i] & 0xFF;
        uint32_t p = 0;
        
        for (uint32_t j = 0; j < 8; j++) {
            if (b & 1) {
                p ^= a;
            }
            uint32_t hi_bit_set = a & 0x80;
            a <<= 1;
            if (hi_bit_set) {
                a ^= 0x1B; // Reduce by irreducible polynomial
            }
            b >>= 1;
        }
        
        output_C[i] = p & 0xFF;
    }
}

// Accelerator implementation of Galois Field GF(2^8) multiplication
LOOM_ACCEL()
void gf_mul_dsa(const uint32_t* __restrict__ input_A,
                const uint32_t* __restrict__ input_B,
                uint32_t* __restrict__ output_C,
                const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t i = 0; i < N; i++) {
        uint32_t a = input_A[i] & 0xFF;
        uint32_t b = input_B[i] & 0xFF;
        uint32_t p = 0;
        
        for (uint32_t j = 0; j < 8; j++) {
            if (b & 1) {
                p ^= a;
            }
            uint32_t hi_bit_set = a & 0x80;
            a <<= 1;
            if (hi_bit_set) {
                a ^= 0x1B; // Reduce by irreducible polynomial
            }
            b >>= 1;
        }
        
        output_C[i] = p & 0xFF;
    }
}



