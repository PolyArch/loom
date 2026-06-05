/* downsample: strided read decimation.
 * Function variant: kernel implemented as a separate function. */

#include <stdint.h>
#include <stdio.h>

#define INPUT_SIZE 16
#define FACTOR 4
#define OUTPUT_SIZE (INPUT_SIZE / FACTOR)
#define EXPECTED_CHECKSUM 250u

__attribute__((noinline))
static void downsample(const float *input, float *output, unsigned output_size, unsigned factor) {
    for (unsigned i = 0; i < output_size; ++i) {
        output[i] = input[i * factor];
    }
}

int main(void) {
    float input[INPUT_SIZE];
    float output[OUTPUT_SIZE];

    for (unsigned i = 0; i < INPUT_SIZE; ++i) {
        input[i] = (float)(i * 3u + 1u);
    }

    downsample(input, output, OUTPUT_SIZE, FACTOR);

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
