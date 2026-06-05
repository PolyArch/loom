/* relu: elementwise float rectified linear activation.
 * Inline variant: kernel loop lives directly in main. */

#include <stdio.h>

#define N 32
#define EXPECTED_CHECKSUM 42.0f

int main(void) {
    float input[N];
    float output[N];

    for (int i = 0; i < N; ++i) {
        input[i] = (float)((i % 13) - 6);
    }

    for (unsigned i = 0; i < N; ++i) {
        float value = input[i];
        output[i] = value > 0.0f ? value : 0.0f;
    }

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
