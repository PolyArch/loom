// Loom kernel implementation: chacha20_qr
#include "chacha20_qr.h"
#include "loom/loom.h"
#include <cmath>
#include <cstdlib>

// Full pipeline test from C++ source: ChaCha20 block function
// Tests complete compilation chain with ARX (add-rotate-xor) operations
// Operates on a 16-word (512-bit) state, applying quarter-round transforms
// for the specified number of double-rounds (ChaCha20 uses 10 double-rounds)

// CPU implementation of ChaCha20 block function
void chacha20_qr_cpu(const uint32_t* __restrict__ input_state,
                     uint32_t* __restrict__ output_state,
                     const uint32_t num_rounds) {
    // Copy input state to output
    for (uint32_t i = 0; i < 16; i++) {
        output_state[i] = input_state[i];
    }

    // Each double-round consists of 4 column quarter-rounds
    // followed by 4 diagonal quarter-rounds
    for (uint32_t round = 0; round < num_rounds; round++) {
        // Column quarter-rounds
        // QR(0, 4,  8, 12)
        output_state[0] += output_state[4];
        output_state[12] ^= output_state[0];
        output_state[12] = (output_state[12] << 16) | (output_state[12] >> 16);
        output_state[8] += output_state[12];
        output_state[4] ^= output_state[8];
        output_state[4] = (output_state[4] << 12) | (output_state[4] >> 20);
        output_state[0] += output_state[4];
        output_state[12] ^= output_state[0];
        output_state[12] = (output_state[12] << 8) | (output_state[12] >> 24);
        output_state[8] += output_state[12];
        output_state[4] ^= output_state[8];
        output_state[4] = (output_state[4] << 7) | (output_state[4] >> 25);

        // QR(1, 5,  9, 13)
        output_state[1] += output_state[5];
        output_state[13] ^= output_state[1];
        output_state[13] = (output_state[13] << 16) | (output_state[13] >> 16);
        output_state[9] += output_state[13];
        output_state[5] ^= output_state[9];
        output_state[5] = (output_state[5] << 12) | (output_state[5] >> 20);
        output_state[1] += output_state[5];
        output_state[13] ^= output_state[1];
        output_state[13] = (output_state[13] << 8) | (output_state[13] >> 24);
        output_state[9] += output_state[13];
        output_state[5] ^= output_state[9];
        output_state[5] = (output_state[5] << 7) | (output_state[5] >> 25);

        // QR(2, 6, 10, 14)
        output_state[2] += output_state[6];
        output_state[14] ^= output_state[2];
        output_state[14] = (output_state[14] << 16) | (output_state[14] >> 16);
        output_state[10] += output_state[14];
        output_state[6] ^= output_state[10];
        output_state[6] = (output_state[6] << 12) | (output_state[6] >> 20);
        output_state[2] += output_state[6];
        output_state[14] ^= output_state[2];
        output_state[14] = (output_state[14] << 8) | (output_state[14] >> 24);
        output_state[10] += output_state[14];
        output_state[6] ^= output_state[10];
        output_state[6] = (output_state[6] << 7) | (output_state[6] >> 25);

        // QR(3, 7, 11, 15)
        output_state[3] += output_state[7];
        output_state[15] ^= output_state[3];
        output_state[15] = (output_state[15] << 16) | (output_state[15] >> 16);
        output_state[11] += output_state[15];
        output_state[7] ^= output_state[11];
        output_state[7] = (output_state[7] << 12) | (output_state[7] >> 20);
        output_state[3] += output_state[7];
        output_state[15] ^= output_state[3];
        output_state[15] = (output_state[15] << 8) | (output_state[15] >> 24);
        output_state[11] += output_state[15];
        output_state[7] ^= output_state[11];
        output_state[7] = (output_state[7] << 7) | (output_state[7] >> 25);

        // Diagonal quarter-rounds
        // QR(0, 5, 10, 15)
        output_state[0] += output_state[5];
        output_state[15] ^= output_state[0];
        output_state[15] = (output_state[15] << 16) | (output_state[15] >> 16);
        output_state[10] += output_state[15];
        output_state[5] ^= output_state[10];
        output_state[5] = (output_state[5] << 12) | (output_state[5] >> 20);
        output_state[0] += output_state[5];
        output_state[15] ^= output_state[0];
        output_state[15] = (output_state[15] << 8) | (output_state[15] >> 24);
        output_state[10] += output_state[15];
        output_state[5] ^= output_state[10];
        output_state[5] = (output_state[5] << 7) | (output_state[5] >> 25);

        // QR(1, 6, 11, 12)
        output_state[1] += output_state[6];
        output_state[12] ^= output_state[1];
        output_state[12] = (output_state[12] << 16) | (output_state[12] >> 16);
        output_state[11] += output_state[12];
        output_state[6] ^= output_state[11];
        output_state[6] = (output_state[6] << 12) | (output_state[6] >> 20);
        output_state[1] += output_state[6];
        output_state[12] ^= output_state[1];
        output_state[12] = (output_state[12] << 8) | (output_state[12] >> 24);
        output_state[11] += output_state[12];
        output_state[6] ^= output_state[11];
        output_state[6] = (output_state[6] << 7) | (output_state[6] >> 25);

        // QR(2, 7,  8, 13)
        output_state[2] += output_state[7];
        output_state[13] ^= output_state[2];
        output_state[13] = (output_state[13] << 16) | (output_state[13] >> 16);
        output_state[8] += output_state[13];
        output_state[7] ^= output_state[8];
        output_state[7] = (output_state[7] << 12) | (output_state[7] >> 20);
        output_state[2] += output_state[7];
        output_state[13] ^= output_state[2];
        output_state[13] = (output_state[13] << 8) | (output_state[13] >> 24);
        output_state[8] += output_state[13];
        output_state[7] ^= output_state[8];
        output_state[7] = (output_state[7] << 7) | (output_state[7] >> 25);

        // QR(3, 4,  9, 14)
        output_state[3] += output_state[4];
        output_state[14] ^= output_state[3];
        output_state[14] = (output_state[14] << 16) | (output_state[14] >> 16);
        output_state[9] += output_state[14];
        output_state[4] ^= output_state[9];
        output_state[4] = (output_state[4] << 12) | (output_state[4] >> 20);
        output_state[3] += output_state[4];
        output_state[14] ^= output_state[3];
        output_state[14] = (output_state[14] << 8) | (output_state[14] >> 24);
        output_state[9] += output_state[14];
        output_state[4] ^= output_state[9];
        output_state[4] = (output_state[4] << 7) | (output_state[4] >> 25);
    }

    // Add original state (ChaCha20 final addition)
    for (uint32_t i = 0; i < 16; i++) {
        output_state[i] += input_state[i];
    }
}

// Accelerator implementation of ChaCha20 block function
LOOM_ACCEL()
void chacha20_qr_dsa(LOOM_MEMORY_BANK(8) const uint32_t* __restrict__ input_state,
                     uint32_t* __restrict__ output_state,
                     const uint32_t num_rounds) {
    // Copy input state to output
    LOOM_NO_PARALLEL
    LOOM_NO_UNROLL
    for (uint32_t i = 0; i < 16; i++) {
        output_state[i] = input_state[i];
    }

    // Each double-round consists of 4 column quarter-rounds
    // followed by 4 diagonal quarter-rounds
    for (uint32_t round = 0; round < num_rounds; round++) {
        // Column quarter-rounds
        // QR(0, 4,  8, 12)
        output_state[0] += output_state[4];
        output_state[12] ^= output_state[0];
        output_state[12] = (output_state[12] << 16) | (output_state[12] >> 16);
        output_state[8] += output_state[12];
        output_state[4] ^= output_state[8];
        output_state[4] = (output_state[4] << 12) | (output_state[4] >> 20);
        output_state[0] += output_state[4];
        output_state[12] ^= output_state[0];
        output_state[12] = (output_state[12] << 8) | (output_state[12] >> 24);
        output_state[8] += output_state[12];
        output_state[4] ^= output_state[8];
        output_state[4] = (output_state[4] << 7) | (output_state[4] >> 25);

        // QR(1, 5,  9, 13)
        output_state[1] += output_state[5];
        output_state[13] ^= output_state[1];
        output_state[13] = (output_state[13] << 16) | (output_state[13] >> 16);
        output_state[9] += output_state[13];
        output_state[5] ^= output_state[9];
        output_state[5] = (output_state[5] << 12) | (output_state[5] >> 20);
        output_state[1] += output_state[5];
        output_state[13] ^= output_state[1];
        output_state[13] = (output_state[13] << 8) | (output_state[13] >> 24);
        output_state[9] += output_state[13];
        output_state[5] ^= output_state[9];
        output_state[5] = (output_state[5] << 7) | (output_state[5] >> 25);

        // QR(2, 6, 10, 14)
        output_state[2] += output_state[6];
        output_state[14] ^= output_state[2];
        output_state[14] = (output_state[14] << 16) | (output_state[14] >> 16);
        output_state[10] += output_state[14];
        output_state[6] ^= output_state[10];
        output_state[6] = (output_state[6] << 12) | (output_state[6] >> 20);
        output_state[2] += output_state[6];
        output_state[14] ^= output_state[2];
        output_state[14] = (output_state[14] << 8) | (output_state[14] >> 24);
        output_state[10] += output_state[14];
        output_state[6] ^= output_state[10];
        output_state[6] = (output_state[6] << 7) | (output_state[6] >> 25);

        // QR(3, 7, 11, 15)
        output_state[3] += output_state[7];
        output_state[15] ^= output_state[3];
        output_state[15] = (output_state[15] << 16) | (output_state[15] >> 16);
        output_state[11] += output_state[15];
        output_state[7] ^= output_state[11];
        output_state[7] = (output_state[7] << 12) | (output_state[7] >> 20);
        output_state[3] += output_state[7];
        output_state[15] ^= output_state[3];
        output_state[15] = (output_state[15] << 8) | (output_state[15] >> 24);
        output_state[11] += output_state[15];
        output_state[7] ^= output_state[11];
        output_state[7] = (output_state[7] << 7) | (output_state[7] >> 25);

        // Diagonal quarter-rounds
        // QR(0, 5, 10, 15)
        output_state[0] += output_state[5];
        output_state[15] ^= output_state[0];
        output_state[15] = (output_state[15] << 16) | (output_state[15] >> 16);
        output_state[10] += output_state[15];
        output_state[5] ^= output_state[10];
        output_state[5] = (output_state[5] << 12) | (output_state[5] >> 20);
        output_state[0] += output_state[5];
        output_state[15] ^= output_state[0];
        output_state[15] = (output_state[15] << 8) | (output_state[15] >> 24);
        output_state[10] += output_state[15];
        output_state[5] ^= output_state[10];
        output_state[5] = (output_state[5] << 7) | (output_state[5] >> 25);

        // QR(1, 6, 11, 12)
        output_state[1] += output_state[6];
        output_state[12] ^= output_state[1];
        output_state[12] = (output_state[12] << 16) | (output_state[12] >> 16);
        output_state[11] += output_state[12];
        output_state[6] ^= output_state[11];
        output_state[6] = (output_state[6] << 12) | (output_state[6] >> 20);
        output_state[1] += output_state[6];
        output_state[12] ^= output_state[1];
        output_state[12] = (output_state[12] << 8) | (output_state[12] >> 24);
        output_state[11] += output_state[12];
        output_state[6] ^= output_state[11];
        output_state[6] = (output_state[6] << 7) | (output_state[6] >> 25);

        // QR(2, 7,  8, 13)
        output_state[2] += output_state[7];
        output_state[13] ^= output_state[2];
        output_state[13] = (output_state[13] << 16) | (output_state[13] >> 16);
        output_state[8] += output_state[13];
        output_state[7] ^= output_state[8];
        output_state[7] = (output_state[7] << 12) | (output_state[7] >> 20);
        output_state[2] += output_state[7];
        output_state[13] ^= output_state[2];
        output_state[13] = (output_state[13] << 8) | (output_state[13] >> 24);
        output_state[8] += output_state[13];
        output_state[7] ^= output_state[8];
        output_state[7] = (output_state[7] << 7) | (output_state[7] >> 25);

        // QR(3, 4,  9, 14)
        output_state[3] += output_state[4];
        output_state[14] ^= output_state[3];
        output_state[14] = (output_state[14] << 16) | (output_state[14] >> 16);
        output_state[9] += output_state[14];
        output_state[4] ^= output_state[9];
        output_state[4] = (output_state[4] << 12) | (output_state[4] >> 20);
        output_state[3] += output_state[4];
        output_state[14] ^= output_state[3];
        output_state[14] = (output_state[14] << 8) | (output_state[14] >> 24);
        output_state[9] += output_state[14];
        output_state[4] ^= output_state[9];
        output_state[4] = (output_state[4] << 7) | (output_state[4] >> 25);
    }

    // Add original state (ChaCha20 final addition)
    LOOM_PARALLEL()
    LOOM_UNROLL()
    for (uint32_t i = 0; i < 16; i++) {
        output_state[i] += input_state[i];
    }
}
