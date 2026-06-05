/* variance: two-pass population variance over fixed float samples.
 * Inline variant: kernel loops live directly in main. */

#include <stdio.h>

#define N 16
#define EXPECTED_VARIANCE 4.214844f

int main(void) {
    float input[N];

    for (int i = 0; i < N; ++i) {
        input[i] = (float)((i % 7) - 3) + 0.25f;
    }

    float sum = 0.0f;
    for (unsigned i = 0; i < N; ++i) {
        sum += input[i];
    }

    float mean = sum / (float)N;
    float var = 0.0f;
    for (unsigned i = 0; i < N; ++i) {
        float diff = input[i] - mean;
        var += diff * diff;
    }

    float result = var / (float)N;
    printf("variance value: %.6f\n", result);
    if (result < EXPECTED_VARIANCE - 0.0001f || result > EXPECTED_VARIANCE + 0.0001f) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
