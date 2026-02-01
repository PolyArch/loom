#include <cstdio>

#include "rle_decode.h"

int main() {
    const uint32_t encoded_length = 7;
    
    // Input RLE encoded data
    uint32_t values[encoded_length] = {1, 2, 3, 4, 5, 6, 7};
    uint32_t counts[encoded_length] = {3, 2, 4, 5, 1, 3, 2};
    
    // Calculate output size
    uint32_t output_size = 0;
    for (uint32_t i = 0; i < encoded_length; i++) {
        output_size += counts[i];
    }
    
    // Output arrays
    uint32_t expect_output[output_size];
    uint32_t calculated_output[output_size];
    
    // Compute expected result with CPU version
    rle_decode_cpu(values, counts, expect_output, encoded_length);
    
    // Compute result with accelerator version
    rle_decode_dsa(values, counts, calculated_output, encoded_length);
    
    // Compare results
    for (uint32_t i = 0; i < output_size; i++) {
        if (expect_output[i] != calculated_output[i]) {
            printf("rle_decode: FAILED\n");
            return 1;
        }
    }
    
    printf("rle_decode: PASSED\n");
    return 0;
}


