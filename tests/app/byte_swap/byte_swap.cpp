// Loom kernel implementation: byte_swap
#include "byte_swap.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: Byte swap (endianness conversion)
// Tests complete compilation chain with bitwise shift and mask operations
// Test: input=[0x12345678,0xAABBCCDD,0x11223344,0xFF00FF00], N=4 → byte-swapped values
// Note: Output is displayed as signed i32, so 0xDDCCBBAA appears as -573785174

// CPU implementation of byte swap (endianness conversion)
// Reverses byte order within each 32-bit value
// 0x12345678 becomes 0x78563412
void byte_swap_cpu(const uint32_t* __restrict__ input_data,
                   uint32_t* __restrict__ output_swapped,
                   const uint32_t N) {
    for (uint32_t i = 0; i < N; i++) {
        uint32_t value = input_data[i];

        uint32_t byte0 = (value >> 0) & 0xFF;
        uint32_t byte1 = (value >> 8) & 0xFF;
        uint32_t byte2 = (value >> 16) & 0xFF;
        uint32_t byte3 = (value >> 24) & 0xFF;

        output_swapped[i] = (byte0 << 24) | (byte1 << 16) | (byte2 << 8) | byte3;
    }
}

// Accelerator implementation of byte swap (endianness conversion)
LOOM_ACCEL()
void byte_swap_dsa(const uint32_t* __restrict__ input_data,
                   uint32_t* __restrict__ output_swapped,
                   const uint32_t N) {
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t i = 0; i < N; i++) {
        uint32_t value = input_data[i];

        uint32_t byte0 = (value >> 0) & 0xFF;
        uint32_t byte1 = (value >> 8) & 0xFF;
        uint32_t byte2 = (value >> 16) & 0xFF;
        uint32_t byte3 = (value >> 24) & 0xFF;

        output_swapped[i] = (byte0 << 24) | (byte1 << 16) | (byte2 << 8) | byte3;
    }
}

