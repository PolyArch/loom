/* xor_block: elementwise 32-bit XOR.
 * Function variant: kernel implemented as a separate function. */

#include <stdint.h>
#include <stdio.h>

#define N 32
#define EXPECTED_CHECKSUM 0xffe05ce0u

__attribute__((noinline))
static void xor_block(const uint32_t *lhs, const uint32_t *rhs, uint32_t *out, unsigned n) {
    for (unsigned i = 0; i < n; ++i) {
        out[i] = lhs[i] ^ rhs[i];
    }
}

int main(void) {
    uint32_t lhs[N];
    uint32_t rhs[N];
    uint32_t out[N];

    for (uint32_t i = 0; i < N; ++i) {
        lhs[i] = 0x12345678u + i * 0x01010101u;
        rhs[i] = 0x0f0f0f0fu ^ (i * 0x11111111u);
    }

    xor_block(lhs, rhs, out, N);

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
