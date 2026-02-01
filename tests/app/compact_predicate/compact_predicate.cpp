// Loom kernel implementation: compact_predicate
#include "compact_predicate.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Predicated stream compaction
// Tests complete compilation chain with predicate-based filtering
// Test: input=[10,20,30,40,50,60,70,80], predicate=[1,0,1,0,1,1,0,1], N=8 → return value=5

// CPU implementation of predicated stream compaction
// Filter elements based on a predicate array (1 = keep, 0 = discard)
// Returns the number of kept elements
uint32_t compact_predicate_cpu(const uint32_t* __restrict__ input,
                               const uint32_t* __restrict__ predicate,
                               uint32_t* __restrict__ output,
                               const uint32_t N) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < N; i++) {
        if (predicate[i] != 0) {
            output[count] = input[i];
            count++;
        }
    }
    return count;
}

// Accelerator implementation of predicated stream compaction
LOOM_ACCEL()
uint32_t compact_predicate_dsa(LOOM_MEMORY_BANK(8) LOOM_STREAM const uint32_t* __restrict__ input,
                               LOOM_STREAM const uint32_t* __restrict__ predicate,
                               uint32_t* __restrict__ output,
                               const uint32_t N) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < N; i++) {
        if (predicate[i] != 0) {
            output[count] = input[i];
            count++;
        }
    }
    return count;
}

