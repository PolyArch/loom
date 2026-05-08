/* dotproduct: float dot product acc = sum_i a[i] * b[i].
 * Inline variant: kernel inlined directly in main. */

#include <stdio.h>

#define N 64

int main(void) {
    float a[N];
    float b[N];

    for (int i = 0; i < N; ++i) {
        a[i] = (float)i;
        b[i] = 1.0f;
    }

    float acc = 0.0f;
    for (int i = 0; i < N; ++i) {
        acc += a[i] * b[i];
    }

    printf("%.6f\n", acc);
    return 0;
}
