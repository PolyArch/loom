/* upsample: zero-insertion expansion.
 * Inline variant: kernel body inlined into main. */

#include <stdint.h>
#include <stdio.h>

#define INPUT_SIZE 4
#define FACTOR 4
#define OUTPUT_SIZE (INPUT_SIZE * FACTOR)
#define EXPECTED_CHECKSUM 242u

int main(void) {
    float input[INPUT_SIZE] = {2.0f, 5.0f, 8.0f, 11.0f};
    float output[OUTPUT_SIZE];

    for (unsigned i = 0; i < OUTPUT_SIZE; ++i) {
        output[i] = 0.0f;
    }

    for (unsigned i = 0; i < INPUT_SIZE; ++i) {
        output[i * FACTOR] = input[i];
    }

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
