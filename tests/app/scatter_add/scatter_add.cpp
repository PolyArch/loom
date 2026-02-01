// Loom kernel implementation: scatter_add
#include "scatter_add.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Scatter-add operation (indirect write with accumulation)
// Tests complete compilation chain with indirect memory access pattern: dst[indices[i]] += src[i]
// Test: src=[15,27,8,42,19,33,11,25], indices=[0,3,1,3,5,2,1,4], dst_init=[5,10,7,3,12,8]
// Result: dst=[20,29,40,72,37,27]
// Note: Indices 1 and 3 appear twice, testing accumulation (dst[1]=10+8+11=29, dst[3]=3+27+42=72)

// CPU implementation of scatter-add operation
// Scatter values from src to dst using indices with atomic add
// dst[indices[i]] += src[i] for all i
// This is the core operation for graph algorithms like BFS and SSSP
void scatter_add_cpu(const uint32_t* __restrict__ src,
                     const uint32_t* __restrict__ indices,
                     uint32_t* __restrict__ dst,
                     const uint32_t N,
                     const uint32_t dst_size) {
    for (uint32_t i = 0; i < N; i++) {
        uint32_t idx = indices[i];
        if (idx < dst_size) {
            dst[idx] += src[i];
        }
    }
}

// Scatter-add: dst[indices[i]] += src[i] (indirect write to dst using indices array, with accumulation)
// Accelerator implementation of scatter-add operation
LOOM_ACCEL()
void scatter_add_dsa(LOOM_MEMORY_BANK(8) LOOM_STREAM const uint32_t* __restrict__ src,
                     LOOM_STREAM const uint32_t* __restrict__ indices,
                     uint32_t* __restrict__ dst,
                     const uint32_t N,
                     const uint32_t dst_size) {
    for (uint32_t i = 0; i < N; i++) {
        uint32_t idx = indices[i];
        if (idx < dst_size) {
            dst[idx] += src[i];
        }
    }
}

// Read src array linearly
// Read indices array linearly
// Scatter-add operation: indirect atomic add using indices

