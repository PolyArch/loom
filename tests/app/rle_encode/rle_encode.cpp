// Loom kernel implementation: rle_encode
#include "rle_encode.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Run-length encoding
// Tests complete compilation chain with stateful compression loop
// Test: [5,5,5,3,3,7] → values=[5,3,7], counts=[3,2,1], length=3






// CPU implementation of run-length encoding
void rle_encode_cpu(const uint32_t* __restrict__ input_data,
                    uint32_t* __restrict__ output_values,
                    uint32_t* __restrict__ output_counts,
                    uint32_t* __restrict__ output_length,
                    const uint32_t N) {
    if (N == 0) {
        *output_length = 0;
        return;
    }
    
    uint32_t write_idx = 0;
    uint32_t current_value = input_data[0];
    uint32_t current_count = 1;
    
    for (uint32_t i = 1; i < N; i++) {
        if (input_data[i] == current_value) {
            current_count++;
        } else {
            output_values[write_idx] = current_value;
            output_counts[write_idx] = current_count;
            write_idx++;
            current_value = input_data[i];
            current_count = 1;
        }
    }
    
    // Write the last run
    output_values[write_idx] = current_value;
    output_counts[write_idx] = current_count;
    write_idx++;
    
    *output_length = write_idx;
}

// Accelerator implementation of run-length encoding
LOOM_ACCEL()
void rle_encode_dsa(LOOM_MEMORY_BANK(8) LOOM_STREAM const uint32_t* __restrict__ input_data,
                    LOOM_STREAM uint32_t* __restrict__ output_values,
                    uint32_t* __restrict__ output_counts,
                    uint32_t* __restrict__ output_length,
                    const uint32_t N) {
    if (N == 0) {
        *output_length = 0;
        return;
    }
    
    uint32_t write_idx = 0;
    uint32_t current_value = input_data[0];
    uint32_t current_count = 1;
    
    for (uint32_t i = 1; i < N; i++) {
        if (input_data[i] == current_value) {
            current_count++;
        } else {
            output_values[write_idx] = current_value;
            output_counts[write_idx] = current_count;
            write_idx++;
            current_value = input_data[i];
            current_count = 1;
        }
    }
    
    // Write the last run
    output_values[write_idx] = current_value;
    output_counts[write_idx] = current_count;
    write_idx++;
    
    *output_length = write_idx;
}



