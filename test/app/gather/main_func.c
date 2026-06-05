/* gather: indirect read with out-of-bounds zero fill.
 * Function variant: kernel implemented as a separate function. */

#include <stdint.h>
#include <stdio.h>

#define N 16
#define SRC_SIZE 10
#define EXPECTED_CHECKSUM 174u

__attribute__((noinline))
static void gather(const uint32_t *src, const uint32_t *indices, uint32_t *dst, unsigned n, unsigned src_size) {
    for (unsigned i = 0; i < n; ++i) {
        uint32_t idx = indices[i];
        dst[i] = idx < src_size ? src[idx] : 0u;
    }
}

int main(void) {
    uint32_t src[SRC_SIZE];
    uint32_t indices[N] = {0u, 3u, 9u, 10u, 2u, 7u, 12u, 1u,
                           5u, 8u, 6u, 4u, 15u, 0u, 9u, 11u};
    uint32_t dst[N];

    for (uint32_t i = 0; i < SRC_SIZE; ++i) {
        src[i] = i * 3u + 1u;
    }

    gather(src, indices, dst, N, SRC_SIZE);

    uint32_t checksum = 0;
    for (unsigned i = 0; i < N; ++i) {
        checksum += dst[i];
    }

    printf("gather checksum: %u\n", checksum);
    if (checksum != EXPECTED_CHECKSUM) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
