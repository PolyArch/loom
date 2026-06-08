/* compact: stable stream compaction of non-zero elements.
 * Inline variant: kernel body inlined into main. */

#include <stdint.h>
#include <stdio.h>

#define N 12
#define EXPECTED_CHECKSUM 1400u
#define EXPECTED_COUNT 7u

int main(void) {
    uint32_t input[N] = {10u, 0u, 20u, 0u, 30u, 40u, 0u, 50u, 0u, 60u, 70u, 0u};
    uint32_t output[N] = {0u};
    uint32_t count = 0;

    for (unsigned i = 0; i < N; ++i) {
        if (input[i] != 0u) {
            output[count] = input[i];
            ++count;
        }
    }

    uint32_t checksum = 0;
    for (uint32_t i = 0; i < count; ++i) {
        checksum += (i + 1u) * output[i];
    }

    printf("compact checksum: %u count: %u\n", checksum, count);
    if (checksum != EXPECTED_CHECKSUM || count != EXPECTED_COUNT) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
