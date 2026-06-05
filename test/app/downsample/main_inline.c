/* downsample: strided read decimation.
 * Inline variant: kernel body inlined into main. */

#include <stdint.h>
#include <stdio.h>

#define INPUT_SIZE 16
#define FACTOR 4
#define OUTPUT_SIZE (INPUT_SIZE / FACTOR)
#define EXPECTED_CHECKSUM 250u

int main(void) {
    float input[INPUT_SIZE];
    float output[OUTPUT_SIZE];

    for (unsigned i = 0; i < INPUT_SIZE; ++i) {
        input[i] = (float)(i * 3u + 1u);
    }

    for (unsigned i = 0; i < OUTPUT_SIZE; ++i) {
        output[i] = input[i * FACTOR];
    }

    uint32_t checksum = 0;
    for (unsigned i = 0; i < OUTPUT_SIZE; ++i) {
        checksum += (uint32_t)(i + 1u) * (uint32_t)output[i];
    }

    printf("downsample checksum: %u\n", checksum);
    if (checksum != EXPECTED_CHECKSUM) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
