/* upsample: zero-insertion expansion.
 * Function variant: kernel implemented as a separate function. */

#include <stdint.h>
#include <stdio.h>

#define INPUT_SIZE 4
#define FACTOR 4
#define OUTPUT_SIZE (INPUT_SIZE * FACTOR)
#define EXPECTED_CHECKSUM 242u

__attribute__((noinline))
static void upsample(const float *input, float *output, unsigned output_size, unsigned factor) {
    for (unsigned i = 0; i < output_size; ++i) {
        output[i] = (i % factor) == 0u ? input[i / factor] : 0.0f;
    }
}

int main(void) {
    float input[INPUT_SIZE] = {2.0f, 5.0f, 8.0f, 11.0f};
    float output[OUTPUT_SIZE];

    upsample(input, output, OUTPUT_SIZE, FACTOR);

    uint32_t checksum = 0;
    for (unsigned i = 0; i < OUTPUT_SIZE; ++i) {
        checksum += (uint32_t)(i + 1u) * (uint32_t)output[i];
    }

    printf("upsample checksum: %u\n", checksum);
    if (checksum != EXPECTED_CHECKSUM) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
