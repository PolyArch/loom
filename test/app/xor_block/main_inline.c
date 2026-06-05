/* xor_block: elementwise 32-bit XOR.
 * Inline variant: kernel loop lives directly in main. */

#include <stdint.h>
#include <stdio.h>

#define N 32
#define EXPECTED_CHECKSUM 0xffe05ce0u

int main(void) {
    uint32_t lhs[N];
    uint32_t rhs[N];
    uint32_t out[N];

    for (uint32_t i = 0; i < N; ++i) {
        lhs[i] = 0x12345678u + i * 0x01010101u;
        rhs[i] = 0x0f0f0f0fu ^ (i * 0x11111111u);
    }

    for (unsigned i = 0; i < N; ++i) {
        out[i] = lhs[i] ^ rhs[i];
    }

    uint32_t checksum = 0;
    for (unsigned i = 0; i < N; ++i) {
        checksum += out[i];
    }

    printf("xor_block checksum: 0x%08x\n", checksum);
    if (checksum != EXPECTED_CHECKSUM) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
