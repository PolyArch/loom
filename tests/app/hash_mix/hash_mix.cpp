// Loom kernel implementation: hash_mix
#include "hash_mix.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Hash mixing operation
// Tests complete compilation chain with bitwise operations (rotate, XOR)
// Test: state=[0,123456789,987654321] + data=[1234567,7654321,1111111] → [3280104067,3325253434,684477807] (in signed form)






// CPU implementation of hash mixing operation
// Performs a simple hash mixing step similar to those in SHA and MD5
// Mix function: rotate left, add, XOR pattern
void hash_mix_cpu(const uint32_t* __restrict__ input_state,
                  const uint32_t* __restrict__ input_data,
                  uint32_t* __restrict__ output_state,
                  const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        uint32_t s = input_state[i];
        uint32_t d = input_data[i];
        
        // Mix operation: typical hash function mixing steps
        s = s + d;
        s = (s << 7) | (s >> 25);  // Rotate left by 7
        s = s ^ d;
        s = s * 0x5BD1E995;        // Mix with constant
        s = (s << 13) | (s >> 19); // Rotate left by 13
        
        output_state[i] = s;
    }
}

// Accelerator implementation of hash mixing operation
LOOM_ACCEL()
void hash_mix_dsa(const uint32_t* __restrict__ input_state,
                  const uint32_t* __restrict__ input_data,
                  uint32_t* __restrict__ output_state,
                  const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL()
    for (uint32_t i = 0; i < N; i++) {
        uint32_t s = input_state[i];
        uint32_t d = input_data[i];
        
        // Mix operation: typical hash function mixing steps
        s = s + d;
        s = (s << 7) | (s >> 25);  // Rotate left by 7
        s = s ^ d;
        s = s * 0x5BD1E995;        // Mix with constant
        s = (s << 13) | (s >> 19); // Rotate left by 13
        
        output_state[i] = s;
    }
}





