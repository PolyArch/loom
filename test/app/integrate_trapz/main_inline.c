/* integrate_trapz: trapezoidal integration over fixed float samples.
 * Inline variant: kernel loop lives directly in main. */

#include <stdio.h>

#define N 9
#define EXPECTED_AREA 0.335938f

int main(void) {
    float x[N];
    float y[N];

    for (int i = 0; i < N; ++i) {
        x[i] = (float)i / (float)(N - 1);
        y[i] = x[i] * x[i];
    }

    float area = 0.0f;
    for (unsigned i = 0; i + 1u < N; ++i) {
        float dx = x[i + 1u] - x[i];
        float avg_y = (y[i] + y[i + 1u]) * 0.5f;
        area += avg_y * dx;
    }

    printf("integrate_trapz area: %.6f\n", area);
    if (area < EXPECTED_AREA - 0.0001f || area > EXPECTED_AREA + 0.0001f) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
