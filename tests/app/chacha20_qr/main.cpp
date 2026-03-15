#include <cstdio>

#include "chacha20_qr.h"

int main() {
    // Standard ChaCha20 test vector (RFC 7539 Section 2.3.2)
    // State is 16 x uint32_t = 512 bits
    // Layout: constants(4) | key(8) | counter(1) | nonce(3)
    uint32_t input_state[16] = {
        0x61707865, 0x3320646e, 0x79622d32, 0x6b206574,  // "expand 32-byte k"
        0x03020100, 0x07060504, 0x0b0a0908, 0x0f0e0d0c,  // key (part 1)
        0x13121110, 0x17161514, 0x1b1a1918, 0x1f1e1d1c,  // key (part 2)
        0x00000001, 0x09000000, 0x4a000000, 0x00000000   // counter | nonce
    };

    // 10 double-rounds for ChaCha20
    const uint32_t num_rounds = 10;

    uint32_t expect_output[16];
    uint32_t calculated_output[16];

    // Compute expected result with CPU version
    chacha20_qr_cpu(input_state, expect_output, num_rounds);

    // Compute result with accelerator version
    chacha20_qr_dsa(input_state, calculated_output, num_rounds);

    // Compare results
    for (uint32_t i = 0; i < 16; i++) {
        if (expect_output[i] != calculated_output[i]) {
            printf("chacha20_qr: FAILED (mismatch at word %u: expected 0x%08x, got 0x%08x)\n",
                   i, expect_output[i], calculated_output[i]);
            return 1;
        }
    }

    printf("chacha20_qr: PASSED\n");
    return 0;
}
