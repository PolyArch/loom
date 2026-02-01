// Loom kernel implementation: bitonic_stage
#include "bitonic_stage.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Bitonic sort stage (in-place)
// Tests complete compilation chain with conditional swapping logic
// Test: inplace=[3,1,4,2,8,6,7,5], N=8, stage=1, pass=0 → inplace=[1,3,2,4,8,6,7,5]






// CPU implementation of bitonic sort stage (in-place)
void bitonic_stage_cpu(float* __restrict__ inplace,
                       const uint32_t N,
                       const uint32_t stage,
                       const uint32_t pass) {
    uint32_t distance = 1U << pass;
    uint32_t block_size = 1U << (stage + 1);

    for (uint32_t i = 0; i < N; i++) {
        uint32_t block_idx = i / block_size;
        uint32_t idx_in_block = i % block_size;

        // Determine if we should compare with forward or backward partner
        uint32_t half_block = block_size / 2;
        uint32_t ascending = (block_idx % 2 == 0) ? 1 : 0;

        // Only process if we're in the first half of comparison pairs
        if ((idx_in_block & distance) == 0) {
            uint32_t partner = i + distance;
            if (partner < N) {
                uint32_t should_swap = 0;
                if (ascending) {
                    should_swap = (inplace[i] > inplace[partner]) ? 1 : 0;
                } else {
                    should_swap = (inplace[i] < inplace[partner]) ? 1 : 0;
                }

                if (should_swap) {
                    float temp = inplace[i];
                    inplace[i] = inplace[partner];
                    inplace[partner] = temp;
                }
            }
        }
    }
}

// Accelerator implementation of bitonic sort stage (in-place)
LOOM_ACCEL()
void bitonic_stage_dsa(float* __restrict__ inplace,
                       const uint32_t N,
                       const uint32_t stage,
                       const uint32_t pass) {
    uint32_t distance = 1U << pass;
    uint32_t block_size = 1U << (stage + 1);

    LOOM_PARALLEL(4, interleaved)
    LOOM_TRIPCOUNT_RANGE(10, 1000)
    for (uint32_t i = 0; i < N; i++) {
        uint32_t block_idx = i / block_size;
        uint32_t idx_in_block = i % block_size;

        // Determine if we should compare with forward or backward partner
        uint32_t half_block = block_size / 2;
        uint32_t ascending = (block_idx % 2 == 0) ? 1 : 0;

        // Only process if we're in the first half of comparison pairs
        if ((idx_in_block & distance) == 0) {
            uint32_t partner = i + distance;
            if (partner < N) {
                uint32_t should_swap = 0;
                if (ascending) {
                    should_swap = (inplace[i] > inplace[partner]) ? 1 : 0;
                } else {
                    should_swap = (inplace[i] < inplace[partner]) ? 1 : 0;
                }

                if (should_swap) {
                    float temp = inplace[i];
                    inplace[i] = inplace[partner];
                    inplace[partner] = temp;
                }
            }
        }
    }
}



