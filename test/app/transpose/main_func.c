/* transpose: row-major matrix transpose.
 * Function variant: kernel implemented as a separate function. */

#include <stdint.h>
#include <stdio.h>

#define M 3
#define N 5
#define EXPECTED_WEIGHTED_CHECKSUM 2080u

__attribute__((noinline))
static void transpose(const uint32_t *input, uint32_t *output, unsigned rows, unsigned cols) {
    for (unsigned i = 0; i < rows; ++i) {
        for (unsigned j = 0; j < cols; ++j) {
            output[j * rows + i] = input[i * cols + j];
        }
    }
}

int main(void) {
    uint32_t input[M * N];
    uint32_t output[M * N];

    for (uint32_t i = 0; i < M * N; ++i) {
        input[i] = i * 2u + 1u;
        output[i] = 0u;
    }

    transpose(input, output, M, N);

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
