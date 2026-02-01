// Loom kernel implementation: delta_decode
#include "delta_decode.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Delta decoding (cumulative sum)
// Tests complete compilation chain with scan operation
// Test: input_deltas=[100,1,2,3,4], N=5 → output=[100,101,103,106,110]






// CPU implementation of delta decoding
void delta_decode_cpu(const uint32_t* __restrict__ input_deltas,
                      uint32_t* __restrict__ output_data,
                      const uint32_t N) {
    if (N == 0) {
        return;
    }
    
    // First element is the base value
    output_data[0] = input_deltas[0];
    
    // Subsequent elements are cumulative sums
    for (uint32_t i = 1; i < N; i++) {
        output_data[i] = output_data[i - 1] + input_deltas[i];
    }
}

// Accelerator implementation of delta decoding
LOOM_ACCEL()
void delta_decode_dsa(const uint32_t* __restrict__ input_deltas,
                      uint32_t* __restrict__ output_data,
                      const uint32_t N) {
    if (N == 0) {
        return;
    }
    
    // First element is the base value
    output_data[0] = input_deltas[0];
    
    // Subsequent elements are cumulative sums
    LOOM_PARALLEL()
    LOOM_UNROLL()
    for (uint32_t i = 1; i < N; i++) {
        output_data[i] = output_data[i - 1] + input_deltas[i];
    }
}



