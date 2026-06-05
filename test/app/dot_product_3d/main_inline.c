/* dot_product_3d: strided 3D float dot products.
 * Inline variant: kernel loop lives directly in main. */

#include <stdio.h>

#define N 16
#define EXPECTED_CHECKSUM 402.0f

int main(void) {
    float a[N * 3];
    float b[N * 3];
    float out[N];

    for (int i = 0; i < N; ++i) {
        a[i * 3 + 0] = (float)(i + 1);
        a[i * 3 + 1] = (float)((i % 5) - 2);
        a[i * 3 + 2] = (float)((i % 3) + 1);
        b[i * 3 + 0] = 2.0f;
        b[i * 3 + 1] = -3.0f;
        b[i * 3 + 2] = 4.0f;
    }

    for (unsigned i = 0; i < N; ++i) {
        float ax = a[i * 3u + 0u];
        float ay = a[i * 3u + 1u];
        float az = a[i * 3u + 2u];
        float bx = b[i * 3u + 0u];
        float by = b[i * 3u + 1u];
        float bz = b[i * 3u + 2u];
        out[i] = ax * bx + ay * by + az * bz;
    }

    float checksum = 0.0f;
    for (unsigned i = 0; i < N; ++i) {
        checksum += out[i];
    }

    printf("dot_product_3d checksum: %.1f\n", checksum);
    if (checksum < EXPECTED_CHECKSUM - 0.5f || checksum > EXPECTED_CHECKSUM + 0.5f) {
        printf("FAILED\n");
        return 1;
    }
    printf("PASSED\n");
    return 0;
}
