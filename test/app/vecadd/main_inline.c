/* vecadd: element-wise float vector add c[i] = a[i] + b[i].
 * Inline variant: kernel inlined directly in main. */

#include <stdio.h>

#define N 64

int main(void) {
    float a[N];
    float b[N];
    float c[N];

    for (int i = 0; i < N; ++i) {
        a[i] = (float)i;
        b[i] = 0.5f * (float)i;
    }

    for (int i = 0; i < N; ++i) {
        c[i] = a[i] + b[i];
    }

    float checksum = 0.0f;
    for (int i = 0; i < N; ++i) {
        checksum += c[i];
    }

    printf("%.6f\n", checksum);
    return 0;
}
