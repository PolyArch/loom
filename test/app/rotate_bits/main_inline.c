/* rotate_bits: elementwise 32-bit left rotate with variable shifts.
 * Inline variant: kernel loop lives directly in main. */

#include <stdint.h>
#include <stdio.h>

#define N 32
#define EXPECTED_CHECKSUM 0x204080efu

int main(void) {
    uint32_t input[N];
    uint32_t shift[N];
    uint32_t out[N];

    for (uint32_t i = 0; i < N; ++i) {
        input[i] = 0x89abcdefu + i * 0x01020408u;
        shift[i] = i;
    }

    for (unsigned i = 0; i < N; ++i) {
        uint32_t amount = shift[i] & 31u;
        uint32_t value = input[i];
        out[i] = amount == 0u ? value : ((value << amount) | (value >> (32u - amount)));
    }

    uint32_t checksum = 0;
    for (unsigned i = 0; i < N; ++i) {
        checksum += out[i];
    }

    printf("rotate_bits checksum: 0x%08x\n", checksum);
    if (checksum != EXPECTED_CHECKSUM) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
