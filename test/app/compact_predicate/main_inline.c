/* compact_predicate: inline stable compaction using a separate predicate array. */

#include <stdint.h>
#include <stdio.h>

#define N 8u
#define EXPECTED_CHECKSUM 860u
#define EXPECTED_COUNT 5u

int main(void) {
    uint32_t input[N] = {10u, 20u, 30u, 40u, 50u, 60u, 70u, 80u};
    uint32_t predicate[N] = {1u, 0u, 1u, 0u, 1u, 1u, 0u, 1u};
    uint32_t output[N] = {0u};
    uint32_t count = 0;

    for (uint32_t i = 0; i < N; ++i) {
        if (predicate[i] != 0u) {
            output[count] = input[i];
            ++count;
        }
    }

    uint32_t checksum = 0;
    for (uint32_t i = 0; i < count; ++i) {
        checksum += (i + 1u) * output[i];
    }

    printf("compact_predicate checksum: %u count: %u\n", checksum, count);
    if (checksum != EXPECTED_CHECKSUM || count != EXPECTED_COUNT) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
