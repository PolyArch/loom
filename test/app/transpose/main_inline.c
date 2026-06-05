/* transpose: row-major matrix transpose.
 * Inline variant: kernel loops live directly in main. */

#include <stdint.h>
#include <stdio.h>

#define M 3
#define N 5
#define EXPECTED_WEIGHTED_CHECKSUM 2080u

int main(void) {
    uint32_t input[M * N];
    uint32_t output[M * N];

    for (uint32_t i = 0; i < M * N; ++i) {
        input[i] = i * 2u + 1u;
        output[i] = 0u;
    }

    for (unsigned i = 0; i < M; ++i) {
        for (unsigned j = 0; j < N; ++j) {
            output[j * M + i] = input[i * N + j];
        }
    }

    uint32_t weighted = 0;
    for (unsigned i = 0; i < M * N; ++i) {
        weighted += output[i] * (i + 1u);
    }

    printf("transpose weighted_checksum: %u\n", weighted);
    if (weighted != EXPECTED_WEIGHTED_CHECKSUM) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
