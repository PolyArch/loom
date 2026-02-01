// Loom kernel implementation: string_hash
#include "string_hash.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Rolling hash (Rabin-Karp)
// Tests complete compilation chain with modular arithmetic and sliding window
// Test: string=[1,2,3,4,5], window=3 → hashes=[98, 39, 81]






// CPU implementation of rolling hash (Rabin-Karp)
// Computes hash values for all windows of size window_size in the input string
// Base: 256 (for character set), Modulus: 101 (prime number)
void string_hash_cpu(const uint32_t* __restrict__ input_str,
                     uint32_t* __restrict__ output_hashes,
                     const uint32_t N,
                     const uint32_t window_size) {
    const uint32_t base = 256;
    const uint32_t modulus = 101;
    
    if (window_size > N) {
        return;
    }
    
    // Compute base^(window_size-1) % modulus
    uint32_t h = 1;
    for (uint32_t i = 0; i < window_size - 1; i++) {
        h = (h * base) % modulus;
    }
    
    // Compute hash for first window
    uint32_t hash_value = 0;
    for (uint32_t i = 0; i < window_size; i++) {
        hash_value = (hash_value * base + input_str[i]) % modulus;
    }
    output_hashes[0] = hash_value;
    
    // Rolling hash for remaining windows
    for (uint32_t i = 1; i <= N - window_size; i++) {
        // Remove leading character, add trailing character
        hash_value = (hash_value + modulus - (input_str[i - 1] * h) % modulus) % modulus;
        hash_value = (hash_value * base + input_str[i + window_size - 1]) % modulus;
        output_hashes[i] = hash_value;
    }
}

// Accelerator implementation of rolling hash (Rabin-Karp)
LOOM_ACCEL()
void string_hash_dsa(const uint32_t* __restrict__ input_str,
                     uint32_t* __restrict__ output_hashes,
                     const uint32_t N,
                     const uint32_t window_size) {
    const uint32_t base = 256;
    const uint32_t modulus = 101;
    
    if (window_size > N) {
        return;
    }
    
    // Compute base^(window_size-1) % modulus
    uint32_t h = 1;
    LOOM_NO_PARALLEL
    LOOM_NO_UNROLL
    for (uint32_t i = 0; i < window_size - 1; i++) {
        h = (h * base) % modulus;
    }
    
    // Compute hash for first window
    uint32_t hash_value = 0;
    for (uint32_t i = 0; i < window_size; i++) {
        hash_value = (hash_value * base + input_str[i]) % modulus;
    }
    output_hashes[0] = hash_value;
    
    // Rolling hash for remaining windows
    for (uint32_t i = 1; i <= N - window_size; i++) {
        // Remove leading character, add trailing character
        hash_value = (hash_value + modulus - (input_str[i - 1] * h) % modulus) % modulus;
        hash_value = (hash_value * base + input_str[i + window_size - 1]) % modulus;
        output_hashes[i] = hash_value;
    }
}



