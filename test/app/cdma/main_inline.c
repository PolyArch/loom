
#include <stdint.h>
#include <stdio.h>

#define N 32u
#define EXPECTED_WEIGHTED_CHECKSUM 36432u

int main(void) {
    uint32_t src[N];
    uint32_t dst[N];

    for (uint32_t i = 0; i < N; ++i) {
        src[i] = i * 3u + 7u;
        dst[i] = 0u;
    }

    for (uint32_t i = 0; i < N; ++i) {
        dst[i] = src[i];
    }

    uint32_t weighted = 0u;
    for (uint32_t i = 0; i < N; ++i) {
        weighted += dst[i] * (i + 1u);
    }

    printf("cdma weighted_checksum: %u\n", weighted);
    if (weighted != EXPECTED_WEIGHTED_CHECKSUM) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
