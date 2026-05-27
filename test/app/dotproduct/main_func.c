/* dotproduct: float dot product acc = sum_i a[i] * b[i].
 * Function variant: kernel implemented as a separate function. */

#include <stdio.h>

#define N 64

__attribute__((noinline))
static float dotproduct(const float *a, const float *b, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc += a[i] * b[i];
    }
    return acc;
}

int main(void) {
    float a[N];
    float b[N];

    for (int i = 0; i < N; ++i) {
        a[i] = (float)i;
        b[i] = 1.0f;
    }

    float acc = dotproduct(a, b, N);
    printf("%.6f\n", acc);
    return 0;
}
