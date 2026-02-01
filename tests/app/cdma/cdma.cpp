// Loom kernel implementation: cdma
#include "cdma.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: Centralized DMA (memory copy)
// Tests complete compilation chain with simple sequential read and write (memcpy-like pattern)
// Test: SRC=[10,20,30,40,50], N=5 → DST=[10,20,30,40,50]







// CPU implementation of centralized DMA (memory copy)
void cdma_cpu(const uint32_t* __restrict__ SRC, 
              uint32_t* __restrict__ DST, 
              const size_t N) {
    for (size_t i = 0; i < N; i++) {
        DST[i] = SRC[i];
    }
}

// CDMA (memcpy): DST[i] = SRC[i] (simple sequential copy)
// Accelerator implementation of centralized DMA (memory copy)
LOOM_ACCEL()
void cdma_dsa(LOOM_MEMORY_BANK(8) LOOM_STREAM const uint32_t* __restrict__ SRC, 
              LOOM_STREAM uint32_t* __restrict__ DST, 
              const size_t N) {
    for (size_t i = 0; i < N; i++) {
        DST[i] = SRC[i];
    }
}





