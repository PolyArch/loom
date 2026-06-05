/* scatter_add: indirect write with accumulation and bounds check.
 * Inline variant: kernel loop lives directly in main. */

#include <stdint.h>
#include <stdio.h>

#define N 16
#define DST_SIZE 8
#define EXPECTED_WEIGHTED_CHECKSUM 360u

int main(void) {
    uint32_t src[N];
    uint32_t indices[N] = {0u, 3u, 1u, 3u, 7u, 8u, 1u, 4u,
                           7u, 2u, 5u, 3u, 12u, 6u, 0u, 7u};
    uint32_t dst[DST_SIZE];

    for (uint32_t i = 0; i < N; ++i) {
        src[i] = (i % 5u) + 1u;
    }
    for (uint32_t i = 0; i < DST_SIZE; ++i) {
        dst[i] = i;
    }

    for (unsigned i = 0; i < N; ++i) {
        uint32_t idx = indices[i];
        if (idx < DST_SIZE) {
            dst[idx] += src[i];
        }
    }

    uint32_t weighted = 0;
    for (unsigned i = 0; i < DST_SIZE; ++i) {
        weighted += dst[i] * (i + 1u);
    }

    printf("scatter_add weighted_checksum: %u\n", weighted);
    if (weighted != EXPECTED_WEIGHTED_CHECKSUM) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
