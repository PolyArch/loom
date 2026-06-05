/* mean: floating-point sum reduction followed by division.
 * Function variant: kernel implemented as a separate function. */

#include <math.h>
#include <stdio.h>

#define N 64

__attribute__((noinline))
static float mean_kernel(const float *input, unsigned n) {
    float sum = 0.0f;
    for (unsigned i = 0; i < n; ++i) {
        sum += input[i];
    }
    return sum / (float)n;
}

int main(void) {
    float input[N];

    for (unsigned i = 0; i < N; ++i) {
        input[i] = (float)(i % 10);
    }

    float result = mean_kernel(input, N);
    if (fabsf(result - 4.3125f) > 1e-5f) {
        puts("FAILED");
        return 1;
    }

    printf("mean result: %.6f\n", result);
    puts("PASSED");
    return 0;
}
