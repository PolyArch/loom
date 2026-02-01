// Loom kernel implementation: histogram
#include "histogram.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Histogram computation
// Tests complete compilation chain with indirect write pattern (hist[input[i]]++) and initialization loop
// Test: input=[0,1,2,1,3,2,1,4,2,0], 5 bins → [2,3,3,1,1]






// CPU implementation of histogram computation
// Count occurrences of each value in the input array
// hist[i] = count of how many times value i appears in input
void histogram_cpu(const uint32_t* __restrict__ input,
                   uint32_t* __restrict__ hist,
                   const uint32_t N,
                   const uint32_t num_bins) {
    // Initialize histogram to zero
    for (uint32_t i = 0; i < num_bins; i++) {
        hist[i] = 0;
    }
    
    // Count occurrences
    for (uint32_t i = 0; i < N; i++) {
        LOOM_REDUCE(+)
        uint32_t value = input[i];
        if (value < num_bins) {
            hist[value]++;
        }
    }
}

// Histogram: hist[input[i]]++ for all i (counting occurrences via indirect increment)
// Accelerator implementation of histogram computation
LOOM_TARGET("temporal")
LOOM_ACCEL()
void histogram_dsa(const uint32_t* __restrict__ input,
                   uint32_t* __restrict__ hist,
                   const uint32_t N,
                   const uint32_t num_bins) {
    // Initialize histogram to zero
    LOOM_PARALLEL(4)
    for (uint32_t i = 0; i < num_bins; i++) {
        hist[i] = 0;
    }
    
    // Count occurrences
    for (uint32_t i = 0; i < N; i++) {
        uint32_t value = input[i];
        if (value < num_bins) {
            hist[value]++;
        }
    }
}



// Step 1: Initialize histogram to zero

// Step 2: Read input values (these become indices into histogram)

// Step 3: Duplicate input values for bounds check and atomic add

// Step 4: Check bounds: value < num_bins

// Step 5: Partition based on bounds check (valid vs out-of-bounds)
// Only valid indices (value < num_bins) go to first output

// Step 6: Generate ones to add at each indexed location

// Step 7: Indirect atomic increment for valid indices only: hist[input[i]]++



