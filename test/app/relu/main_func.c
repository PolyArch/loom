/* relu: elementwise float rectified linear activation.
 * Function variant: kernel implemented as a separate function. */

#include <stdio.h>

#define N 32
#define EXPECTED_CHECKSUM 42.0f

__attribute__((noinline))
static void relu(const float *input, float *output, unsigned n) {
    for (unsigned i = 0; i < n; ++i) {
        float value = input[i];
        output[i] = value > 0.0f ? value : 0.0f;
    }
}

int main(void) {
    float input[N];
    float output[N];

    for (int i = 0; i < N; ++i) {
        input[i] = (float)((i % 13) - 6);
    }

    relu(input, output, N);

    float checksum = 0.0f;
    for (unsigned i = 0; i < N; ++i) {
        checksum += output[i];
    }

    printf("relu checksum: %.1f\n", checksum);
    if (checksum < EXPECTED_CHECKSUM - 0.25f || checksum > EXPECTED_CHECKSUM + 0.25f) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
