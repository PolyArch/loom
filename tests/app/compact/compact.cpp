// Loom kernel implementation: compact
#include "compact.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Stream compaction
// Tests complete compilation chain with conditional filtering and partition operation
// Test: input=[10,0,20,0,30,40,0,50], N=8 → return value=5






// CPU implementation of stream compaction
// Filter and compact non-zero elements from input to output
// Returns the number of non-zero elements
uint32_t compact_cpu(const uint32_t* __restrict__ input,
                     uint32_t* __restrict__ output,
                     const uint32_t N) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < N; i++) {
        if (input[i] != 0) {
            output[count] = input[i];
            count++;
        }
    }
    return count;
}

// Accelerator implementation of stream compaction
LOOM_ACCEL()
uint32_t compact_dsa(LOOM_MEMORY_BANK(8) LOOM_STREAM const uint32_t* __restrict__ input,
                     LOOM_STREAM uint32_t* __restrict__ output,
                     const uint32_t N) {
    uint32_t count = 0;
    for (uint32_t i = 0; i < N; i++) {
        if (input[i] != 0) {
            output[count] = input[i];
            count++;
        }
    }
    return count;
}



// Step 1: Read input array

// Step 2: Check which elements are non-zero: input[i] != 0

// Step 3: Partition dataflow based on non-zero predicate
// First output (non-zero elements) goes to output, second output (zeros) is discarded

// Step 4: Write compacted non-zero elements to output

// Step 5: Count the number of non-zero elements using count operation




