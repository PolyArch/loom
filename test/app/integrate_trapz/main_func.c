/* integrate_trapz: trapezoidal integration over fixed float samples.
 * Function variant: kernel implemented as a separate function. */

#include <stdio.h>

#define N 9
#define EXPECTED_AREA 0.335938f

__attribute__((noinline))
static float integrate_trapz(const float *x, const float *y, unsigned n) {
    float sum = 0.0f;
    for (unsigned i = 0; i + 1u < n; ++i) {
        float dx = x[i + 1u] - x[i];
        float avg_y = (y[i] + y[i + 1u]) * 0.5f;
        sum += avg_y * dx;
    }
    return sum;
}

int main(void) {
    float x[N];
    float y[N];

    for (int i = 0; i < N; ++i) {
        x[i] = (float)i / (float)(N - 1);
        y[i] = x[i] * x[i];
    }

    float area = integrate_trapz(x, y, N);
    printf("integrate_trapz area: %.6f\n", area);
    if (area < EXPECTED_AREA - 0.0001f || area > EXPECTED_AREA + 0.0001f) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
