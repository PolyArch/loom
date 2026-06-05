/* parity: odd-parity bit reduction for 32-bit words.
 * Function variant: kernel implemented as a separate function. */

#include <stdint.h>
#include <stdio.h>

#define N 32
#define EXPECTED_ODD_COUNT 11u

__attribute__((noinline))
static void parity(const uint32_t *input, uint32_t *out, unsigned n) {
    for (unsigned i = 0; i < n; ++i) {
        uint32_t value = input[i];
        uint32_t bit = 0;
        while (value != 0u) {
            bit ^= value & 1u;
            value >>= 1;
        }
        out[i] = bit;
    }
}

int main(void) {
    uint32_t input[N];
    uint32_t out[N];

    for (uint32_t i = 0; i < N; ++i) {
        if (i == 0u) {
            input[i] = 0u;
        } else if (i == 1u) {
            input[i] = 1u;
        } else if (i == 2u) {
            input[i] = 3u;
        } else if (i == 3u) {
            input[i] = 7u;
        } else {
            input[i] = 0x9abcdef0u * i;
        }
    }

    parity(input, out, N);

    uint32_t odd_count = 0;
    for (unsigned i = 0; i < N; ++i) {
        odd_count += out[i];
    }

    printf("parity odd_count: %u\n", odd_count);
    if (odd_count != EXPECTED_ODD_COUNT) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
