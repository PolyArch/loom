// Loom kernel implementation: modexp
#include "modexp.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Modular exponentiation (a^b mod n)
// Tests complete compilation chain with bit manipulation and while loop
// Test: 2^3 mod 7 = 1, 3^4 mod 7 = 4, 5^2 mod 7 = 4






// CPU implementation of modular exponentiation (a^b mod n)
void modexp_cpu(const uint32_t* __restrict__ input_base,
                const uint32_t* __restrict__ input_exp,
                uint32_t* __restrict__ output_result,
                const uint32_t modulus,
                const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        uint64_t result = 1;
        uint64_t base = input_base[i] % modulus;
        uint32_t exp = input_exp[i];
        
        while (exp > 0) {
            if (exp & 1) {
                result = (result * base) % modulus;
            }
            base = (base * base) % modulus;
            exp >>= 1;
        }
        
        output_result[i] = (uint32_t)result;
    }
}

// Accelerator implementation of modular exponentiation (a^b mod n)
LOOM_ACCEL()
void modexp_dsa(const uint32_t* __restrict__ input_base,
                const uint32_t* __restrict__ input_exp,
                uint32_t* __restrict__ output_result,
                const uint32_t modulus,
                const uint32_t N) {
    LOOM_NO_PARALLEL
    LOOM_NO_UNROLL
    for (uint32_t i = 0; i < N; i++) {
        uint64_t result = 1;
        uint64_t base = input_base[i] % modulus;
        uint32_t exp = input_exp[i];
        
        while (exp > 0) {
            if (exp & 1) {
                result = (result * base) % modulus;
            }
            base = (base * base) % modulus;
            exp >>= 1;
        }
        
        output_result[i] = (uint32_t)result;
    }
}



