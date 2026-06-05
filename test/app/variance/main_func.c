/* variance: two-pass population variance over fixed float samples.
 * Function variant: kernel implemented as a separate function. */

#include <stdio.h>

#define N 16
#define EXPECTED_VARIANCE 4.214844f

__attribute__((noinline))
static float variance(const float *input, unsigned n) {
    float sum = 0.0f;
    for (unsigned i = 0; i < n; ++i) {
        sum += input[i];
    }

    float mean = sum / (float)n;
    float var = 0.0f;
    for (unsigned i = 0; i < n; ++i) {
        float diff = input[i] - mean;
        var += diff * diff;
    }

    return var / (float)n;
}

int main(void) {
    float input[N];

    for (int i = 0; i < N; ++i) {
        input[i] = (float)((i % 7) - 3) + 0.25f;
    }

    float result = variance(input, N);
    printf("variance value: %.6f\n", result);
    if (result < EXPECTED_VARIANCE - 0.0001f || result > EXPECTED_VARIANCE + 0.0001f) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
