/* merge: sorted two-input merge with counted-loop conditional selection.
 * Inline variant: kernel body inlined into main. */

#include <stdint.h>
#include <stdio.h>

#define A_SIZE 5
#define B_SIZE 6
#define OUT_SIZE (A_SIZE + B_SIZE)
#define EXPECTED_CHECKSUM 970u

int main(void) {
    float a[A_SIZE] = {1.0f, 4.0f, 9.0f, 13.0f, 21.0f};
    float b[B_SIZE] = {2.0f, 3.0f, 10.0f, 14.0f, 20.0f, 22.0f};
    float out[OUT_SIZE];
    unsigned i = 0;
    unsigned j = 0;

    for (unsigned k = 0; k < OUT_SIZE; ++k) {
        int take_a = (j >= B_SIZE) || (i < A_SIZE && a[i] <= b[j]);
        if (take_a) {
            out[k] = a[i];
            ++i;
        } else {
            out[k] = b[j];
            ++j;
        }
    }

    uint32_t checksum = 0;
    for (unsigned pos = 0; pos < OUT_SIZE; ++pos) {
        checksum += (uint32_t)(pos + 1u) * (uint32_t)out[pos];
    }

    printf("merge checksum: %u\n", checksum);
    if (checksum != EXPECTED_CHECKSUM) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
