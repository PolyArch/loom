/* partition: stable two-pass partition around a pivot.
 * Inline variant: kernel body inlined into main. */

#include <stdint.h>
#include <stdio.h>

#define N 10
#define EXPECTED_CHECKSUM 371u
#define EXPECTED_PIVOT 5u

int main(void) {
    float input[N] = {3.0f, 7.0f, 1.0f, 9.0f, 5.0f, 2.0f, 8.0f, 4.0f, 6.0f, 10.0f};
    float output[N];
    uint32_t pivot_index = 0;
    uint32_t write_pos = 0;

    for (unsigned i = 0; i < N; ++i) {
        if (input[i] <= 5.5f) {
            output[write_pos] = input[i];
            ++write_pos;
        }
    }

    pivot_index = write_pos;

    for (unsigned i = 0; i < N; ++i) {
        if (input[i] > 5.5f) {
            output[write_pos] = input[i];
            ++write_pos;
        }
    }

    uint32_t checksum = 0;
    for (unsigned i = 0; i < N; ++i) {
        checksum += (uint32_t)(i + 1u) * (uint32_t)output[i];
    }

    printf("partition checksum: %u pivot: %u\n", checksum, pivot_index);
    if (checksum != EXPECTED_CHECKSUM || pivot_index != EXPECTED_PIVOT) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
