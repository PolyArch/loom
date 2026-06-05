/* downsample_avg: averaged decimation over fixed windows.
 * Inline variant: kernel body inlined into main. */

#include <stdint.h>
#include <stdio.h>

#define INPUT_SIZE 16
#define FACTOR 4
#define OUTPUT_SIZE (INPUT_SIZE / FACTOR)
#define EXPECTED_CHECKSUM2X 590u

int main(void) {
    float input[INPUT_SIZE];
    float output[OUTPUT_SIZE];

    for (unsigned i = 0; i < INPUT_SIZE; ++i) {
        input[i] = (float)(i * 3u + 1u);
    }

    for (unsigned i = 0; i < OUTPUT_SIZE; ++i) {
        float sum = 0.0f;
        for (unsigned j = 0; j < FACTOR; ++j) {
            sum += input[i * FACTOR + j];
        }
        output[i] = sum / (float)FACTOR;
    }

    uint32_t checksum2x = 0;
    for (unsigned i = 0; i < OUTPUT_SIZE; ++i) {
        checksum2x += (uint32_t)(i + 1u) * (uint32_t)(output[i] * 2.0f);
    }

    printf("downsample_avg checksum2x: %u\n", checksum2x);
    if (checksum2x != EXPECTED_CHECKSUM2X) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
