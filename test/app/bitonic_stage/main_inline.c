/* bitonic_stage: one in-place compare/swap stage from bitonic sort.
 * Inline variant: kernel body inlined into main. */

#include <stdint.h>
#include <stdio.h>

#define N 8
#define EXPECTED_CHECKSUM 194u

int main(void) {
    float data[N] = {3.0f, 1.0f, 4.0f, 2.0f, 8.0f, 6.0f, 7.0f, 5.0f};
    unsigned distance = 1u;
    unsigned block_size = 4u;
    unsigned pairs = N / 2u;

    for (unsigned pair = 0; pair < pairs; ++pair) {
        unsigned i = pair * 2u;
        unsigned partner = i + distance;
        unsigned block_idx = i / block_size;
        unsigned ascending = (block_idx % 2u) == 0u;
        float left = data[i];
        float right = data[partner];
        int swap_up = left > right;
        int swap_down = left < right;
        int should_swap = ascending ? swap_up : swap_down;

        data[i] = should_swap ? right : left;
        data[partner] = should_swap ? left : right;
    }

    uint32_t checksum = 0;
    for (unsigned i = 0; i < N; ++i) {
        checksum += (uint32_t)(i + 1u) * (uint32_t)data[i];
    }

    printf("bitonic_stage checksum: %u\n", checksum);
    if (checksum != EXPECTED_CHECKSUM) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
