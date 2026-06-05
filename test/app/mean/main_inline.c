/* mean: floating-point sum reduction followed by division.
 * Inline variant: kernel loop written directly in main. */

#include <math.h>
#include <stdio.h>

#define N 64

int main(void) {
    float input[N];

    for (unsigned i = 0; i < N; ++i) {
        input[i] = (float)(i % 10);
    }

    float sum = 0.0f;
    for (unsigned i = 0; i < N; ++i) {
        sum += input[i];
    }
    float result = sum / (float)N;

    if (fabsf(result - 4.3125f) > 1e-5f) {
        puts("FAILED");
        return 1;
    }

    printf("mean result: %.6f\n", result);
    puts("PASSED");
    return 0;
}
