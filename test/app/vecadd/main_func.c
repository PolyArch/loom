/* vecadd: element-wise float vector add c[i] = a[i] + b[i].
 * Function variant: kernel implemented as a separate function. */

#include <stdio.h>

#define N 64
#define EXPECTED_CHECKSUM 3024.0f

__attribute__((noinline))
static void vecadd(const float *a, const float *b, float *c, int n) {
    for (int i = 0; i < n; ++i) {
        c[i] = a[i] + b[i];
    }
}

static float reduce_sum(const float *x, int n) {
    float acc = 0.0f;
    for (int i = 0; i < n; ++i) {
        acc += x[i];
    }
    return acc;
}

int main(void) {
    float a[N];
    float b[N];
    float c[N];

    for (int i = 0; i < N; ++i) {
        a[i] = (float)i;
        b[i] = 0.5f * (float)i;
    }

    vecadd(a, b, c, N);

    float checksum = reduce_sum(c, N);
    printf("%.6f\n", checksum);
    return checksum == EXPECTED_CHECKSUM ? 0 : 1;
}
