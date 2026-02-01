// Loom kernel implementation: find_first_set
#include "find_first_set.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Find first set bit
// Tests complete compilation chain with conditionals and while loop
// Test: input=[1,2,3,4,5,6,7,8,0] → positions=[1,2,1,3,1,2,1,4,0]






// CPU implementation of find first set (ffs)
// Returns the position of the first (least significant) set bit
// Bit positions are 1-indexed (LSB is position 1)
// Returns 0 if no bits are set
void find_first_set_cpu(const uint32_t* __restrict__ input_data,
                        uint32_t* __restrict__ output_position,
                        const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        uint32_t value = input_data[i];
        
        if (value == 0) {
            output_position[i] = 0;
        } else {
            uint32_t position = 1;
            
            while ((value & 1) == 0) {
                position++;
                value >>= 1;
            }
            
            output_position[i] = position;
        }
    }
}

// Accelerator implementation of find first set (ffs)
LOOM_ACCEL()
void find_first_set_dsa(const uint32_t* __restrict__ input_data,
                        uint32_t* __restrict__ output_position,
                        const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL()
    for (uint32_t i = 0; i < N; i++) {
        uint32_t value = input_data[i];
        
        if (value == 0) {
            output_position[i] = 0;
        } else {
            uint32_t position = 1;
            
            while ((value & 1) == 0) {
                position++;
                value >>= 1;
            }
            
            output_position[i] = position;
        }
    }
}





