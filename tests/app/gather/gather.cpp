// Loom kernel implementation: gather
#include "gather.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Gather operation (indirect read)
// Tests complete compilation chain with indirect memory access pattern: dst[i] = src[indices[i]]
// Then tests simulation: src=[10,20,30,40,50,60,70,80,90], indices=[0,2,4,8,1,10,3,5], N=8, src_size=9
// Expected: dst=[10,30,50,90,20,0,40,60] (index 10 is out of bounds → 0)

// CPU implementation of gather operation
// Gather values from src to dst using indices
// dst[i] = src[indices[i]] for all i
// This is the core operation for graph algorithms that need to collect neighbor information
void gather_cpu(const uint32_t* __restrict__ src,
                const uint32_t* __restrict__ indices,
                uint32_t* __restrict__ dst,
                const uint32_t N,
                const uint32_t src_size) {
    for (uint32_t i = 0; i < N; i++) {
        uint32_t idx = indices[i];
        if (idx < src_size) {
            dst[i] = src[idx];
        } else {
            dst[i] = 0;  // Out of bounds, set to 0
        }
    }
}

// Gather: dst[i] = src[indices[i]] (indirect read from src using indices array)
// Accelerator implementation of gather operation
LOOM_ACCEL()
void gather_dsa(LOOM_MEMORY_BANK(8) LOOM_STREAM const uint32_t* __restrict__ src,
                LOOM_STREAM const uint32_t* __restrict__ indices,
                uint32_t* __restrict__ dst,
                const uint32_t N,
                const uint32_t src_size) {
    for (uint32_t i = 0; i < N; i++) {
        uint32_t idx = indices[i];
        if (idx < src_size) {
            dst[i] = src[idx];
        } else {
            dst[i] = 0;
        }
    }
}

