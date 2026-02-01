// Loom kernel implementation: rle_decode
#include "rle_decode.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Run-length decoding
// Tests complete compilation chain with nested loops and variable-count expansion
// Test: values=[5,3,7], counts=[3,2,1] → [5,5,5,3,3,7]

// CPU implementation of run-length decoding
void rle_decode_cpu(const uint32_t* __restrict__ input_values,
                    const uint32_t* __restrict__ input_counts,
                    uint32_t* __restrict__ output_data,
                    const uint32_t encoded_length) {
    uint32_t write_idx = 0;

    for (uint32_t i = 0; i < encoded_length; i++) {
        uint32_t value = input_values[i];
        uint32_t count = input_counts[i];

        for (uint32_t j = 0; j < count; j++) {
            output_data[write_idx] = value;
            write_idx++;
        }
    }
}

// Accelerator implementation of run-length decoding
LOOM_ACCEL()
void rle_decode_dsa(LOOM_MEMORY_BANK(8) LOOM_STREAM const uint32_t* __restrict__ input_values,
                    LOOM_STREAM const uint32_t* __restrict__ input_counts,
                    uint32_t* __restrict__ output_data,
                    const uint32_t encoded_length) {
    uint32_t write_idx = 0;

    for (uint32_t i = 0; i < encoded_length; i++) {
        uint32_t value = input_values[i];
        uint32_t count = input_counts[i];

        for (uint32_t j = 0; j < count; j++) {
            output_data[write_idx] = value;
            write_idx++;
        }
    }
}

