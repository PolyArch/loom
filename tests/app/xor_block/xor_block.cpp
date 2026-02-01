// Loom kernel implementation: xor_block
#include "xor_block.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// CPU implementation of block-wise XOR
void xor_block_cpu(const uint32_t* __restrict__ input_A,
                   const uint32_t* __restrict__ input_B,
                   uint32_t* __restrict__ output_C,
                   const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        output_C[i] = input_A[i] ^ input_B[i];
    }
}

// Accelerator implementation of block-wise XOR
LOOM_ACCEL()
void xor_block_dsa(const uint32_t* __restrict__ input_A,
                   const uint32_t* __restrict__ input_B,
                   uint32_t* __restrict__ output_C,
                   const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t i = 0; i < N; i++) {
        output_C[i] = input_A[i] ^ input_B[i];
    }
}

