/* merge: sorted two-input merge with counted-loop conditional selection.
 * Function variant: kernel implemented as a separate function. */

#include <stdint.h>
#include <stdio.h>

#define A_SIZE 5
#define B_SIZE 6
#define OUT_SIZE (A_SIZE + B_SIZE)
#define EXPECTED_CHECKSUM 970u

__attribute__((noinline))
static void merge(const float *a, const float *b, float *out, unsigned a_size, unsigned b_size) {
    unsigned i = 0;
    unsigned j = 0;

    for (unsigned k = 0; k < a_size + b_size; ++k) {
        int take_a = (j >= b_size) || (i < a_size && a[i] <= b[j]);
        if (take_a) {
            out[k] = a[i];
            ++i;
        } else {
            out[k] = b[j];
            ++j;
        }
    }
}

int main(void) {
    float a[A_SIZE] = {1.0f, 4.0f, 9.0f, 13.0f, 21.0f};
    float b[B_SIZE] = {2.0f, 3.0f, 10.0f, 14.0f, 20.0f, 22.0f};
    float out[OUT_SIZE];

    merge(a, b, out, A_SIZE, B_SIZE);

    uint32_t checksum = 0;
    for (unsigned i = 0; i < OUT_SIZE; ++i) {
        checksum += (uint32_t)(i + 1u) * (uint32_t)out[i];
    }

    printf("merge checksum: %u\n", checksum);
    if (checksum != EXPECTED_CHECKSUM) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
