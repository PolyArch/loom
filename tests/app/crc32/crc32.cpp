// Loom kernel implementation: crc32
#include "crc32.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>


// Full pipeline test from C++ source: CRC32 checksum
// Tests complete compilation chain with nested loops and bitwise operations
// Test: input=[0x12345678,0xABCDEF00], N=2 → checksum=1119308902
// Note: LLVM optimization generates a 256-element CRC lookup table as a global constant.
// The simulator loads module-level memref.global constants and maps them to handshake.memory
// operations via the dsa.global_memref attribute.






// CPU implementation of CRC32 checksum
void crc32_cpu(const uint32_t* __restrict__ input_data,
               uint32_t* __restrict__ output_checksum,
               const uint32_t N) {
    const uint32_t polynomial = 0xEDB88320;
    uint32_t crc = 0xFFFFFFFF;
    
    for (uint32_t i = 0; i < N; i++) {
        uint32_t data = input_data[i];
        
        // Process all 4 bytes of the uint32_t
        for (uint32_t byte_idx = 0; byte_idx < 4; byte_idx++) {
            uint32_t byte = (data >> (byte_idx * 8)) & 0xFF;
            crc ^= byte;
            
            for (uint32_t bit = 0; bit < 8; bit++) {
                if (crc & 1) {
                    crc = (crc >> 1) ^ polynomial;
                } else {
                    crc = crc >> 1;
                }
            }
        }
    }
    
    *output_checksum = ~crc;
}

// Accelerator implementation of CRC32 checksum
LOOM_ACCEL()
void crc32_dsa(const uint32_t* __restrict__ input_data,
               uint32_t* __restrict__ output_checksum,
               const uint32_t N) {
    const uint32_t polynomial = 0xEDB88320;
    uint32_t crc = 0xFFFFFFFF;
    
    LOOM_PARALLEL()
    LOOM_UNROLL(8)
    for (uint32_t i = 0; i < N; i++) {
        uint32_t data = input_data[i];
        
        // Process all 4 bytes of the uint32_t
        for (uint32_t byte_idx = 0; byte_idx < 4; byte_idx++) {
            uint32_t byte = (data >> (byte_idx * 8)) & 0xFF;
            crc ^= byte;
            
            for (uint32_t bit = 0; bit < 8; bit++) {
                if (crc & 1) {
                    crc = (crc >> 1) ^ polynomial;
                } else {
                    crc = crc >> 1;
                }
            }
        }
    }
    
    *output_checksum = ~crc;
}





